"""EXIF metadata utilities."""

import io
import json
from typing import Optional
from PIL import Image, PngImagePlugin

try:
    import piexif
    PIEXIF_AVAILABLE = True
except ImportError:
    PIEXIF_AVAILABLE = False


def add_exif_metadata(
    image_bytes: bytes,
    output_format: str,
    prompt: str,
    model_name: str,
    seed: int,
    steps: int,
    width: int,
    height: int,
    guidance_scale: float,
) -> bytes:
    """
    Add EXIF metadata to an image indicating AI generation.

    Args:
        image_bytes: Raw image bytes
        output_format: Image format (png, jpeg, webp)
        prompt: Generation prompt
        model_name: Model used for generation
        seed: Random seed used
        steps: Number of inference steps
        width: Image width
        height: Image height
        guidance_scale: CFG scale used

    Returns:
        Image bytes with metadata embedded
    """
    generation_info = {
        "model": model_name,
        "seed": seed,
        "steps": steps,
        "width": width,
        "height": height,
        "guidance_scale": guidance_scale,
        "prompt": prompt[:500],  # Truncate prompt for metadata
    }

    if output_format == "png":
        return _add_png_metadata(image_bytes, prompt, generation_info)
    elif output_format == "jpeg" and PIEXIF_AVAILABLE:
        return _add_jpeg_exif(image_bytes, prompt, generation_info)
    elif output_format == "webp":
        return _add_webp_metadata(image_bytes, prompt, generation_info)

    # Return original if format not supported
    return image_bytes


def _add_png_metadata(
    image_bytes: bytes,
    prompt: str,
    generation_info: dict,
) -> bytes:
    """Add metadata to PNG image using tEXt chunks."""
    image = Image.open(io.BytesIO(image_bytes))

    # Create PNG metadata
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("Software", "Scratchy Image Generator")
    pnginfo.add_text("Comment", "AI Generated")
    pnginfo.add_text("Description", prompt[:500])
    pnginfo.add_text("Generation-Info", json.dumps(generation_info))

    # Save with metadata
    output = io.BytesIO()
    image.save(output, format="PNG", pnginfo=pnginfo)
    output.seek(0)
    return output.getvalue()


def _add_jpeg_exif(
    image_bytes: bytes,
    prompt: str,
    generation_info: dict,
) -> bytes:
    """Add EXIF metadata to JPEG image."""
    if not PIEXIF_AVAILABLE:
        return image_bytes

    image = Image.open(io.BytesIO(image_bytes))

    # Create EXIF data
    exif_dict = {
        "0th": {
            piexif.ImageIFD.Software: "Scratchy Image Generator",
            piexif.ImageIFD.ImageDescription: prompt[:500],
        },
        "Exif": {
            piexif.ExifIFD.UserComment: f"AI Generated with {generation_info['model']}, seed={generation_info['seed']}, steps={generation_info['steps']}".encode(),
        },
    }

    exif_bytes = piexif.dump(exif_dict)

    # Save with EXIF
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=95, exif=exif_bytes)
    output.seek(0)
    return output.getvalue()


def _add_webp_metadata(
    image_bytes: bytes,
    prompt: str,
    generation_info: dict,
) -> bytes:
    """Add metadata to WebP image using XMP."""
    image = Image.open(io.BytesIO(image_bytes))

    # WebP supports EXIF through PIL, but XMP is more reliable
    # For simplicity, we'll add metadata as EXIF if piexif is available
    if PIEXIF_AVAILABLE:
        exif_dict = {
            "0th": {
                piexif.ImageIFD.Software: "Scratchy Image Generator",
                piexif.ImageIFD.ImageDescription: prompt[:500],
            },
            "Exif": {
                piexif.ExifIFD.UserComment: f"AI Generated".encode(),
            },
        }
        exif_bytes = piexif.dump(exif_dict)

        output = io.BytesIO()
        image.save(output, format="WEBP", quality=90, exif=exif_bytes)
        output.seek(0)
        return output.getvalue()

    return image_bytes


def strip_metadata(image_bytes: bytes, output_format: str) -> bytes:
    """Remove all metadata from an image."""
    image = Image.open(io.BytesIO(image_bytes))

    # Create a new image without metadata
    data = list(image.getdata())
    new_image = Image.new(image.mode, image.size)
    new_image.putdata(data)

    output = io.BytesIO()
    format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
    new_image.save(output, format=format_map.get(output_format, "PNG"))
    output.seek(0)
    return output.getvalue()
