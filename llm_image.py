"""
Shared helper: turn raw image bytes into an Anthropic vision content block.

Downscaling the long edge (config.LLM_IMAGE_MAX_DIM) before base64-encoding is
the main lever for keeping vision calls token-friendly — image input tokens
scale with pixel area. Everything is re-encoded to JPEG to shrink the payload.
"""
import base64
import io

from PIL import Image

import config


def encode_image_block(image_bytes: bytes, max_dim: int | None = None) -> dict:
    """Return an Anthropic image block: {"type": "image", "source": {...}}."""
    max_dim = max_dim or config.LLM_IMAGE_MAX_DIM
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != "RGB":
        img = img.convert("RGB")

    long_edge = max(img.size)
    if long_edge > max_dim:
        scale = max_dim / long_edge
        img = img.resize(
            (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
        )

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    data = base64.standard_b64encode(buf.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": data},
    }
