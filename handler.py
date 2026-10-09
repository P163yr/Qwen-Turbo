import os
import io
import gc
import json
import base64
import traceback

import torch
import runpod

from PIL import Image

# ============================================================
# TRANSFORMERS = PROMPT ENHANCER
# ============================================================

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

# ============================================================
# DIFFUSERS = IMAGE GENERATOR
# ============================================================

from diffusers import QwenImage21Pipeline


# ============================================================
# PATHS
# ============================================================

TURBO_MODEL_PATH = os.environ.get(
    "TURBO_MODEL_PATH",
    "/runpod-volume/comfyui/models/diffusers/Qwen-Image-2.1-Turbo"
)

PE_I2I_MODEL_PATH = os.environ.get(
    "PE_I2I_MODEL_PATH",
    "/runpod-volume/comfyui/models/prompt_extenders/Qwen-Image-2.1-PE-I2I"
)


DEFAULT_WIDTH = int(
    os.environ.get("DEFAULT_WIDTH", "1024")
)

DEFAULT_HEIGHT = int(
    os.environ.get("DEFAULT_HEIGHT", "1024")
)

DEFAULT_OUTPUT_RESOLUTION = int(
    os.environ.get("DEFAULT_OUTPUT_RESOLUTION", "1024")
)

DEFAULT_ENHANCE_PROMPT = (
    os.environ.get("DEFAULT_ENHANCE_PROMPT", "true").lower()
    == "true"
)


# ============================================================
# GLOBALS
# ============================================================

turbo_pipe = None

pe_model = None
pe_processor = None
pe_system_prompt = None


# ============================================================
# LOGGING
# ============================================================

def log(message):
    print(message, flush=True)


# ============================================================
# CUDA INFO
# ============================================================

log("=" * 80)
log("Qwen Image 2.1 Turbo + PE-I2I Worker")
log("=" * 80)

log(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required.")

gpu = torch.cuda.get_device_properties(0)

log(f"GPU: {gpu.name}")
log(
    f"VRAM: "
    f"{gpu.total_memory / (1024 ** 3):.2f} GB"
)

log(f"PyTorch: {torch.__version__}")
log(f"CUDA: {torch.version.cuda}")


# ============================================================
# VERIFY TURBO PATH
# ============================================================

log("=" * 80)
log(f"Turbo model: {TURBO_MODEL_PATH}")
log(
    f"Turbo exists: "
    f"{os.path.isdir(TURBO_MODEL_PATH)}"
)

if not os.path.isdir(TURBO_MODEL_PATH):
    raise RuntimeError(
        f"Turbo model directory missing: "
        f"{TURBO_MODEL_PATH}"
    )


# ============================================================
# VERIFY PE PATH
# ============================================================

log(f"PE-I2I model: {PE_I2I_MODEL_PATH}")
log(
    f"PE-I2I exists: "
    f"{os.path.isdir(PE_I2I_MODEL_PATH)}"
)

PE_AVAILABLE = os.path.isdir(
    PE_I2I_MODEL_PATH
)

if PE_AVAILABLE:

    system_prompt_file = os.path.join(
        PE_I2I_MODEL_PATH,
        "system_prompt.txt"
    )

    if not os.path.isfile(system_prompt_file):

        log(
            "WARNING: system_prompt.txt missing. "
            "Prompt enhancement disabled."
        )

        PE_AVAILABLE = False

else:

    log(
        "PE-I2I model not found. "
        "I2I requests will use raw prompts."
    )


# ============================================================
# LOAD TURBO
# ============================================================

log("=" * 80)
log("Loading Qwen Image 2.1 Turbo...")

turbo_pipe = QwenImage21Pipeline.from_pretrained(
    TURBO_MODEL_PATH,
    dtype=torch.bfloat16,
    local_files_only=True,
    low_cpu_mem_usage=True,
)

# 24GB GPU:
# Keep inactive Turbo components in CPU RAM and move each
# component onto CUDA only when required.
turbo_pipe.enable_model_cpu_offload()


# ------------------------------------------------------------
# VAE memory optimizations
# ------------------------------------------------------------

try:
    turbo_pipe.vae.enable_tiling()
    log("VAE tiling enabled.")
except Exception as e:
    log(f"VAE tiling unavailable: {e}")

try:
    turbo_pipe.vae.enable_slicing()
    log("VAE slicing enabled.")
except Exception as e:
    log(f"VAE slicing unavailable: {e}")


log("Turbo loaded.")
log(
    "Checkpoint's built-in Turbo sampling schedule "
    "will be used."
)


# ============================================================
# LOAD PE-I2I LAZILY
#
# THIS IS WHERE THE TRANSFORMERS LOADER IS USED.
# ============================================================

def load_pe_model():

    global pe_model
    global pe_processor
    global pe_system_prompt

    if not PE_AVAILABLE:
        return False

    if pe_model is not None:
        return True

    log("=" * 80)
    log("Loading Qwen Image 2.1 PE-I2I...")
    log(f"Path: {PE_I2I_MODEL_PATH}")


    # ========================================================
    # TRANSFORMERS LOADER #1
    #
    # Loads tokenizer + multimodal image processor
    # ========================================================

    pe_processor = AutoProcessor.from_pretrained(
        PE_I2I_MODEL_PATH,
        local_files_only=True,
    )


    # ========================================================
    # TRANSFORMERS LOADER #2
    #
    # Loads Qwen3.5-VL 9B prompt-enhancement model.
    #
    # Initially stays in CPU RAM.
    # We move it to CUDA only while rewriting the prompt.
    # ========================================================

    pe_model = (
        AutoModelForImageTextToText
        .from_pretrained(
            PE_I2I_MODEL_PATH,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        .eval()
    )


    # Keep PE model on CPU until needed.
    pe_model.to("cpu")


    # --------------------------------------------------------
    # Official system prompt supplied with PE-I2I
    # --------------------------------------------------------

    system_prompt_path = os.path.join(
        PE_I2I_MODEL_PATH,
        "system_prompt.txt"
    )

    with open(
        system_prompt_path,
        "r",
        encoding="utf-8",
    ) as f:

        pe_system_prompt = f.read().strip()


    log("PE-I2I loaded into CPU RAM.")

    return True


# ============================================================
# BASE64 IMAGE HELPERS
# ============================================================

def decode_base64_image(value):

    if not isinstance(value, str):

        raise ValueError(
            "Image must be a base64 string."
        )


    if value.startswith("data:"):

        if "," not in value:

            raise ValueError(
                "Invalid base64 data URL."
            )

        value = value.split(",", 1)[1]


    try:

        raw = base64.b64decode(value)

    except Exception as e:

        raise ValueError(
            f"Invalid base64 data: {e}"
        )


    try:

        image = Image.open(
            io.BytesIO(raw)
        )

        image.load()

        return image.convert("RGB")

    except Exception as e:

        raise ValueError(
            f"Invalid image data: {e}"
        )


def normalize_images(items):

    if not items:
        return None

    if not isinstance(items, list):

        raise ValueError(
            "'images' must be an array."
        )


    result = []


    for index, item in enumerate(items):

        if isinstance(item, str):

            encoded = item

        elif isinstance(item, dict):

            encoded = item.get("image")

            if not encoded:

                raise ValueError(
                    f"images[{index}] is missing "
                    "'image'."
                )

        else:

            raise ValueError(
                f"images[{index}] has invalid format."
            )


        result.append(
            decode_base64_image(encoded)
        )


    return result


def encode_png(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG"
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return (
        "data:image/png;base64,"
        + encoded
    )


# ============================================================
# EXTRACT PE RESULT
# ============================================================

def parse_pe_result(generated_text):

    # Official PE output:
    #
    # <think>
    # ...
    # </think>
    # {
    #   "rewritten_prompt": "...",
    #   "wh_ratio": "",
    #   "ratio_follow": "<image1>"
    # }

    _, separator, answer = generated_text.partition(
        "</think>"
    )

    if separator:
        answer = answer.strip()
    else:
        answer = generated_text.strip()


    # Some generation variants could wrap JSON in markdown.
    answer = answer.replace(
        "```json",
        ""
    ).replace(
        "```",
        ""
    ).strip()


    try:

        result = json.loads(answer)

    except Exception as e:

        log(
            "PE JSON parsing failed."
        )

        log(
            f"Raw PE output: "
            f"{generated_text[:3000]}"
        )

        raise RuntimeError(
            f"Prompt enhancer returned "
            f"invalid JSON: {e}"
        )


    rewritten = result.get(
        "rewritten_prompt"
    )


    if not rewritten:

        raise RuntimeError(
            "Prompt enhancer did not return "
            "'rewritten_prompt'."
        )


    return result


# ============================================================
# PROMPT ENHANCEMENT
# ============================================================

def enhance_i2i_prompt(
    raw_prompt,
    images,
    max_new_tokens=1024,
):

    if not images:

        # This endpoint currently only has PE-I2I.
        return {
            "rewritten_prompt": raw_prompt,
            "wh_ratio": "",
            "ratio_follow": "",
        }


    if not load_pe_model():

        return {
            "rewritten_prompt": raw_prompt,
            "wh_ratio": "",
            "ratio_follow": "",
        }


    log("=" * 80)
    log("Running PE-I2I prompt enhancement...")


    # --------------------------------------------------------
    # Ensure as much GPU memory as possible is free.
    # --------------------------------------------------------

    gc.collect()
    torch.cuda.empty_cache()


    # --------------------------------------------------------
    # Temporarily move Qwen3.5-VL PE model to GPU.
    # --------------------------------------------------------

    log("Moving PE-I2I to GPU...")

    pe_model.to("cuda")


    # --------------------------------------------------------
    # Build official-style multimodal message
    # --------------------------------------------------------

    user_content = []


    # Image order matters:
    #
    # first image  = <image1>
    # second image = <image2>
    # etc.
    for image in images:

        user_content.append(
            {
                "type": "image",
                "image": image,
            }
        )


    user_content.append(
        {
            "type": "text",
            "text": raw_prompt,
        }
    )


    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": pe_system_prompt,
                }
            ],
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]


    # ========================================================
    # THIS IS THE TRANSFORMERS PROCESSOR
    #
    # Converts text + PIL images into Qwen3.5-VL tensors.
    # ========================================================

    inputs = pe_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=True,
    ).to("cuda")


    # ========================================================
    # THIS IS THE TRANSFORMERS GENERATION CALL
    #
    # The prompt enhancer generates the rewritten instruction.
    # ========================================================

    with torch.no_grad():

        output = pe_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
        )


    generated_text = (
        pe_processor.tokenizer.decode(
            output[
                0,
                inputs["input_ids"].shape[1]:
            ],
            skip_special_tokens=True,
        )
    )


    log(
        f"PE raw output: "
        f"{generated_text[:2000]}"
    )


    result = parse_pe_result(
        generated_text
    )


    log(
        "Enhanced prompt: "
        + result["rewritten_prompt"][:2000]
    )


    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Move 9B PE model back to CPU BEFORE Turbo generation.
    #
    # Otherwise both models would compete for ~24GB VRAM.
    # --------------------------------------------------------

    log("Moving PE-I2I back to CPU...")

    pe_model.to("cpu")


    del inputs
    del output

    gc.collect()
    torch.cuda.empty_cache()


    return result


# ============================================================
# TURBO GENERATION
# ============================================================

def generate(inp):

    raw_prompt = inp.get("prompt")

    if not raw_prompt:

        raise ValueError(
            "'prompt' is required."
        )


    raw_prompt = raw_prompt.strip()


    seed = int(
        inp.get("seed", 42)
    )


    images = normalize_images(
        inp.get("images")
    )


    use_kv_cache = bool(
        inp.get(
            "use_kv_cache",
            True
        )
    )


    enhance_prompt = bool(
        inp.get(
            "enhance_prompt",
            DEFAULT_ENHANCE_PROMPT
        )
    )


    # ========================================================
    # PROMPT ENHANCEMENT
    # ========================================================

    pe_result = None

    final_prompt = raw_prompt

    prompt_enhanced = False


    if images and enhance_prompt:

        pe_result = enhance_i2i_prompt(
            raw_prompt,
            images,
            max_new_tokens=int(
                inp.get(
                    "pe_max_new_tokens",
                    1024
                )
            ),
        )

        final_prompt = pe_result[
            "rewritten_prompt"
        ]

        prompt_enhanced = (
            final_prompt != raw_prompt
        )


    elif not images and enhance_prompt:

        # We currently have only PE-I2I.
        #
        # T2I goes directly into Turbo.
        log(
            "T2I request: no PE-T2I model installed. "
            "Using raw prompt."
        )


    # ========================================================
    # GENERATOR
    # ========================================================

    generator = torch.Generator(
        device="cuda"
    ).manual_seed(seed)


    # ========================================================
    # TURBO INPUTS
    # ========================================================

    kwargs = {
        "prompt": final_prompt,
        "generator": generator,
        "use_kv_cache": use_kv_cache,
    }


    # --------------------------------------------------------
    # I2I / MULTI-IMAGE
    # --------------------------------------------------------

    if images:

        kwargs["image"] = images

        kwargs["output_resolution"] = int(
            inp.get(
                "output_resolution",
                DEFAULT_OUTPUT_RESOLUTION,
            )
        )


        if inp.get("width") is not None:

            kwargs["width"] = int(
                inp["width"]
            )


        if inp.get("height") is not None:

            kwargs["height"] = int(
                inp["height"]
            )


    # --------------------------------------------------------
    # T2I
    # --------------------------------------------------------

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
    # DO NOT ADD:
    #
    # num_inference_steps
    # sigmas
    #
    # Qwen Image 2.1 Turbo has its intended schedule stored
    # with the checkpoint.
    # ========================================================


    log("=" * 80)
    log("Running Qwen Image 2.1 Turbo")
    log(
        f"Mode: "
        f"{'I2I' if images else 'T2I'}"
    )
    log(f"Seed: {seed}")
    log(f"Enhanced: {prompt_enhanced}")
    log(f"KV cache: {use_kv_cache}")
    log(f"Final prompt: {final_prompt[:2000]}")


    gc.collect()
    torch.cuda.empty_cache()


    # ========================================================
    # DIFFUSERS IMAGE GENERATION
    # ========================================================

    with torch.inference_mode():

        result = turbo_pipe(
            **kwargs
        )


    output_image = result.images[0]


    gc.collect()
    torch.cuda.empty_cache()


    response = {
        "image": encode_png(
            output_image
        ),
        "seed": seed,
        "mode": (
            "image_edit"
            if images
            else "text_to_image"
        ),
        "prompt_enhanced": prompt_enhanced,
        "raw_prompt": raw_prompt,
        "prompt_used": final_prompt,
        "width": output_image.width,
        "height": output_image.height,
        "turbo_steps": 8,
        "kv_cache": use_kv_cache,
    }


    if pe_result:

        response[
            "pe_wh_ratio"
        ] = pe_result.get(
            "wh_ratio",
            ""
        )

        response[
            "pe_ratio_follow"
        ] = pe_result.get(
            "ratio_follow",
            ""
        )


    return response


# ============================================================
# RUNPOD
# ============================================================

def handler(job):

    try:

        inp = job.get(
            "input",
            {}
        )

        return generate(inp)

    except Exception as e:

        log("=" * 80)
        log("ERROR")
        log(str(e))
        log(traceback.format_exc())
        log("=" * 80)

        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


log("=" * 80)
log("Starting RunPod handler...")
log("=" * 80)

runpod.serverless.start(
    {
        "handler": handler
    }
)
