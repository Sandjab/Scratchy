"""Input sanitization utilities."""

import re
import unicodedata
from typing import Tuple


def sanitize_prompt(prompt: str, max_length: int = 2000) -> str:
    """
    Sanitize a prompt for safe processing.

    - Strips control characters
    - Normalizes Unicode
    - Truncates to max length

    Args:
        prompt: The input prompt
        max_length: Maximum allowed length

    Returns:
        Sanitized prompt
    """
    if not prompt:
        return ""

    # Normalize Unicode (NFC form)
    prompt = unicodedata.normalize("NFC", prompt)

    # Remove control characters except newlines and tabs
    prompt = "".join(
        char for char in prompt
        if unicodedata.category(char) not in ("Cc", "Cf")
        or char in ("\n", "\t")
    )

    # Collapse multiple spaces/newlines
    prompt = re.sub(r"[ \t]+", " ", prompt)
    prompt = re.sub(r"\n{3,}", "\n\n", prompt)

    # Strip leading/trailing whitespace
    prompt = prompt.strip()

    # Truncate to max length
    if len(prompt) > max_length:
        prompt = prompt[:max_length]

    return prompt


def validate_dimensions(
    width: int,
    height: int,
    min_dim: int = 256,
    max_dim: int = 2048,
    multiple: int = 64,
) -> Tuple[int, int, list[str]]:
    """
    Validate and adjust image dimensions.

    Args:
        width: Requested width
        height: Requested height
        min_dim: Minimum dimension
        max_dim: Maximum dimension
        multiple: Dimensions must be multiples of this

    Returns:
        Tuple of (adjusted_width, adjusted_height, warnings)
    """
    warnings = []

    # Clamp to valid range
    if width < min_dim:
        warnings.append(f"width adjusted from {width} to {min_dim} (minimum)")
        width = min_dim
    elif width > max_dim:
        warnings.append(f"width adjusted from {width} to {max_dim} (maximum)")
        width = max_dim

    if height < min_dim:
        warnings.append(f"height adjusted from {height} to {min_dim} (minimum)")
        height = min_dim
    elif height > max_dim:
        warnings.append(f"height adjusted from {height} to {max_dim} (maximum)")
        height = max_dim

    # Round to multiple
    def round_to_multiple(value: int) -> int:
        return ((value + multiple // 2) // multiple) * multiple

    orig_width, orig_height = width, height
    width = round_to_multiple(width)
    height = round_to_multiple(height)

    if width != orig_width:
        warnings.append(f"width adjusted from {orig_width} to {width} (multiple of {multiple})")
    if height != orig_height:
        warnings.append(f"height adjusted from {orig_height} to {height} (multiple of {multiple})")

    return width, height, warnings


def is_safe_filename(filename: str) -> bool:
    """Check if a filename is safe (no path traversal)."""
    if not filename:
        return False

    # Check for path traversal
    if ".." in filename or "/" in filename or "\\" in filename:
        return False

    # Check for null bytes
    if "\x00" in filename:
        return False

    return True
