import os
import io
import base64
import torch
import runpod

from PIL import Image
from diffusers import QwenImage21Pipeline


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/runpod-volume/comfyui/models/diffusers/Qwen-Image-2.1-Turbo"
)


# ============================================================
# LOAD MODEL ONCE PER WORKER
# ============================================================

print("=" * 70)
print("Loading Qwen Image 2.1 Turbo")
print("Model path:", MODEL_PATH)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print("GPU:", props.name)
    print("VRAM GB:", round(props.total_memory / 1024**3, 2))

print("=" * 70)


pipe = QwenImage21Pipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True,
)


# Qwen Image 2.1 Turbo BF16 is too large to keep every component
# resident simultaneously on a typical RTX 4090 24GB.
#
# Model CPU offload keeps the active module on GPU while moving
# inactive components back to system RAM.
pipe.enable_model_cpu_offload()

# Helps larger image decoding without requiring huge VAE memory.
pipe.vae.enable_tiling()

print("Qwen Image 2.1 Turbo loaded successfully.")


# ============================================================
# IMAGE HELPERS
# ============================================================

def decode_base64_image(value):
    """
    Accepts:
      data:image/png;base64,...
      data:image/jpeg;base64,...
      raw base64
    """

    if not isinstance(value, str):
        raise ValueError("Image must be a base64 string.")

    if "," in value and value.startswith("data:"):
        value = value.split(",", 1)[1]

    image_bytes = base64.b64decode(value)

    return Image.open(
        io.BytesIO(image_bytes)
    ).convert("RGBA")


def encode_image(image):
    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
        optimize=False
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return "data:image/png;base64," + encoded


def normalize_input_images(items):
    """
    Supports:

    "images": [
        "data:image/png;base64,..."
    ]

    OR your old RunPod style:

    "images": [
        {
            "name": "input.png",
            "image": "data:image/png;base64,..."
        }
    ]
    """

    if not items:
        return None

    output = []

    for item in items:

        if isinstance(item, dict):
            value = item.get("image")

            if not value:
                raise ValueError(
                    "Image object must contain an 'image' field."
                )

        elif isinstance(item, str):
            value = item

        else:
            raise ValueError(
                "Each image must be a base64 string or an object containing 'image'."
            )

        output.append(
            decode_base64_image(value)
        )

    return output


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):

    inp = job.get("input", {})

    prompt = inp.get("prompt")

    if not prompt:
        return {
            "error": "Missing required input: prompt"
        }

    seed = int(
        inp.get("seed", 0)
    )

    images = normalize_input_images(
        inp.get("images")
    )

    use_kv_cache = bool(
        inp.get("use_kv_cache", True)
    )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    # ========================================================
    # BASE PIPELINE PARAMETERS
    #
    # IMPORTANT:
    # DO NOT manually specify num_inference_steps.
    #
    # Qwen Image 2.1 Turbo contains its official 8-step
    # sampling schedule in the checkpoint.
    # ========================================================

    kwargs = {
        "prompt": prompt,
        "generator": generator,
        "use_kv_cache": use_kv_cache,
    }


    # ========================================================
    # IMAGE EDITING / MULTI-REFERENCE
    # ========================================================

    if images:

        kwargs["image"] = images

        output_resolution = int(
            inp.get("output_resolution", 1024)
        )

        kwargs["output_resolution"] = output_resolution

        # Optional explicit output size.
        #
        # If omitted, Qwen automatically follows the
        # reference-image aspect ratio.

        width = inp.get("width")
        height = inp.get("height")

        if width is not None:
            kwargs["width"] = int(width)

        if height is not None:
            kwargs["height"] = int(height)


    # ========================================================
    # TEXT TO IMAGE
    # ========================================================

    else:

        width = int(
            inp.get("width", 1024)
        )

        height = int(
            inp.get("height", 1024)
        )

        kwargs["width"] = width
        kwargs["height"] = height


    # ========================================================
    # GENERATE
    # ========================================================

    with torch.inference_mode():

        result = pipe(
            **kwargs
        )

    output_image = result.images[0]

    return {
        "image": encode_image(output_image),
        "seed": seed,
        "mode": "image_edit" if images else "text_to_image",
        "turbo_steps": 8,
        "kv_cache": use_kv_cache,
    }


runpod.serverless.start(
    {
        "handler": handler
    }
)
