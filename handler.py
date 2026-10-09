import os
import io
import gc
import base64
import traceback

import torch
import runpod
from PIL import Image

from diffusers import QwenImage21Pipeline


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    "/runpod-volume/comfyui/models/diffusers/Qwen-Image-2.1-Turbo"
)

DEFAULT_WIDTH = int(os.environ.get("DEFAULT_WIDTH", "1024"))
DEFAULT_HEIGHT = int(os.environ.get("DEFAULT_HEIGHT", "1024"))
DEFAULT_OUTPUT_RESOLUTION = int(
    os.environ.get("DEFAULT_OUTPUT_RESOLUTION", "1024")
)


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)


# ============================================================
# CHECK MODEL
# ============================================================

log("=" * 80)
log("Qwen Image 2.1 Turbo Serverless Worker")
log("=" * 80)

log(f"MODEL_PATH: {MODEL_PATH}")
log(f"MODEL_PATH exists: {os.path.exists(MODEL_PATH)}")
log(f"MODEL_PATH is directory: {os.path.isdir(MODEL_PATH)}")

if not os.path.isdir(MODEL_PATH):
    log("ERROR: Turbo model directory does not exist.")

    if os.path.exists("/runpod-volume"):
        log(
            "/runpod-volume contents: "
            + str(os.listdir("/runpod-volume")[:50])
        )

    if os.path.exists("/runpod-volume/comfyui"):
        log(
            "/runpod-volume/comfyui contents: "
            + str(os.listdir("/runpod-volume/comfyui")[:50])
        )

    if os.path.exists("/runpod-volume/comfyui/models"):
        log(
            "/runpod-volume/comfyui/models contents: "
            + str(os.listdir("/runpod-volume/comfyui/models")[:50])
        )

    if os.path.exists("/runpod-volume/comfyui/models/diffusers"):
        log(
            "/runpod-volume/comfyui/models/diffusers contents: "
            + str(
                os.listdir(
                    "/runpod-volume/comfyui/models/diffusers"
                )[:50]
            )
        )

    raise RuntimeError(
        f"Model directory not found: {MODEL_PATH}"
    )

log("Turbo model files:")
for filename in os.listdir(MODEL_PATH)[:50]:
    log(f"  - {filename}")


# ============================================================
# CUDA INFO
# ============================================================

log("=" * 80)
log(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

gpu = torch.cuda.get_device_properties(0)

TOTAL_VRAM_GB = gpu.total_memory / (1024 ** 3)

log(f"GPU: {gpu.name}")
log(f"VRAM: {TOTAL_VRAM_GB:.2f} GB")
log(f"PyTorch: {torch.__version__}")
log(f"CUDA: {torch.version.cuda}")
log("=" * 80)


# ============================================================
# LOAD TURBO PIPELINE
# ============================================================

log("Loading Qwen Image 2.1 Turbo...")

pipe = QwenImage21Pipeline.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True
)


# ============================================================
# MEMORY MANAGEMENT
# ============================================================

# Your current GPU exposes ~24 GB.
#
# Keeping the entire BF16 Turbo pipeline resident is unsafe at
# this VRAM level, so use Accelerate model CPU offload.
#
# Active modules are transferred onto the GPU automatically.

log("Enabling model CPU offload...")

pipe.enable_model_cpu_offload()

# Reduce VAE memory spikes.
try:
    pipe.vae.enable_tiling()
    log("VAE tiling enabled.")
except Exception as e:
    log(f"VAE tiling unavailable: {e}")

try:
    pipe.vae.enable_slicing()
    log("VAE slicing enabled.")
except Exception as e:
    log(f"VAE slicing unavailable: {e}")


log("=" * 80)
log("Qwen Image 2.1 Turbo loaded successfully.")
log("Official Turbo 8-step schedule will be used automatically.")
log("=" * 80)


# ============================================================
# IMAGE FUNCTIONS
# ============================================================

def decode_base64_image(value):
    """
    Supports:

    data:image/png;base64,...
    data:image/jpeg;base64,...
    raw base64
    """

    if not isinstance(value, str):
        raise ValueError("Image must be a base64 string.")

    if value.startswith("data:"):
        if "," not in value:
            raise ValueError("Invalid data URL.")

        value = value.split(",", 1)[1]

    try:
        raw = base64.b64decode(value)
    except Exception as e:
        raise ValueError(
            f"Invalid base64 image: {e}"
        )

    try:
        image = Image.open(io.BytesIO(raw))
        image.load()
    except Exception as e:
        raise ValueError(
            f"Unable to decode image: {e}"
        )

    # Qwen pipeline handles conversion internally,
    # but normalize here for predictable API behaviour.
    if image.mode not in ("RGB", "RGBA"):
        image = image.convert("RGB")

    return image


def normalize_images(items):
    """
    Accepts:

    "images": [
        "data:image/png;base64,..."
    ]

    OR

    "images": [
        {
            "name": "reference.png",
            "image": "data:image/png;base64,..."
        }
    ]
    """

    if not items:
        return None

    if not isinstance(items, list):
        raise ValueError(
            "'images' must be an array."
        )

    images = []

    for index, item in enumerate(items):

        if isinstance(item, str):
            value = item

        elif isinstance(item, dict):
            value = item.get("image")

            if not value:
                raise ValueError(
                    f"images[{index}] does not contain an 'image' field."
                )

        else:
            raise ValueError(
                f"images[{index}] must be a base64 string "
                "or an object containing an 'image' field."
            )

        images.append(
            decode_base64_image(value)
        )

    return images


def encode_png(image):
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


# ============================================================
# GENERATION
# ============================================================

def generate(inp):

    prompt = inp.get("prompt")

    if not prompt or not isinstance(prompt, str):
        raise ValueError(
            "'prompt' is required and must be a string."
        )

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "'prompt' cannot be empty."
        )

    seed = int(
        inp.get("seed", 42)
    )

    use_kv_cache = bool(
        inp.get("use_kv_cache", True)
    )

    images = normalize_images(
        inp.get("images")
    )

    mode = (
        "image_edit"
        if images
        else "text_to_image"
    )


    # --------------------------------------------------------
    # Generator
    # --------------------------------------------------------

    # CPU generator works reliably with model CPU offload.
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)


    # --------------------------------------------------------
    # Base Qwen arguments
    # --------------------------------------------------------

    kwargs = {
        "prompt": prompt,
        "generator": generator,
        "use_kv_cache": use_kv_cache,
    }


    # ========================================================
    # IMAGE EDITING / MULTI REFERENCE
    # ========================================================

    if images:

        kwargs["image"] = images

        output_resolution = int(
            inp.get(
                "output_resolution",
                DEFAULT_OUTPUT_RESOLUTION
            )
        )

        kwargs["output_resolution"] = output_resolution

        # Width / height are optional for image editing.
        #
        # If omitted, Qwen derives output dimensions from
        # the condition-image aspect ratio.

        if inp.get("width") is not None:
            kwargs["width"] = int(
                inp["width"]
            )

        if inp.get("height") is not None:
            kwargs["height"] = int(
                inp["height"]
            )


    # ========================================================
    # TEXT TO IMAGE
    # ========================================================

    else:

        kwargs["width"] = int(
            inp.get(
                "width",
                DEFAULT_WIDTH
            )
        )

        kwargs["height"] = int(
            inp.get(
                "height",
                DEFAULT_HEIGHT
            )
        )


    # ========================================================
    # IMPORTANT:
    #
    # DO NOT add:
    #
    # num_inference_steps=8
    # sigmas=[...]
    #
    # Qwen-Image-2.1-Turbo already has its official sampling
    # sigmas saved in the checkpoint.
    #
    # Diffusers automatically uses them.
    # ========================================================

    log("-" * 80)
    log(f"Mode: {mode}")
    log(f"Seed: {seed}")
    log(f"KV cache: {use_kv_cache}")
    log(f"Prompt: {prompt[:500]}")

    if images:
        log(f"Reference images: {len(images)}")
        log(
            f"Output resolution: "
            f"{kwargs.get('output_resolution')}"
        )
    else:
        log(
            f"Output size: "
            f"{kwargs['width']}x{kwargs['height']}"
        )

    log("Starting Turbo generation...")


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.inference_mode():

        result = pipe(
            **kwargs
        )


    image = result.images[0]

    log(
        f"Generation finished. "
        f"Output size: {image.width}x{image.height}"
    )

    return {
        "image": encode_png(image),
        "seed": seed,
        "mode": mode,
        "width": image.width,
        "height": image.height,
        "turbo_steps": 8,
        "kv_cache": use_kv_cache,
    }


# ============================================================
# RUNPOD HANDLER
# ============================================================

def handler(job):

    try:

        inp = job.get("input", {})

        result = generate(inp)

        return result

    except Exception as e:

        log("=" * 80)
        log("GENERATION ERROR")
        log(str(e))
        log(traceback.format_exc())
        log("=" * 80)

        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }

    finally:

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ============================================================
# SERVERLESS START
# ============================================================

log("Starting RunPod Serverless handler...")

runpod.serverless.start(
    {
        "handler": handler
    }
)
