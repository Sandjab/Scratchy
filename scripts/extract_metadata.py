"""Extract generation metadata from an image into a JSON sidecar.

Supports:
  * AUTOMATIC1111 / Forge style PNG `parameters` text chunk.
  * ComfyUI `prompt` / `workflow` JSON chunks (passed through as-is).
  * EXIF UserComment fallback (some converters store params there).

Usage:
    python scripts/extract_metadata.py <image>
    python scripts/extract_metadata.py <image> --out-dir out/
    python scripts/extract_metadata.py <image> --no-txt

Outputs (alongside the image by default):
    <name>.json   — structured metadata (always written if anything found)
    <name>.txt    — raw `parameters` chunk verbatim (A1111 interop, unless --no-txt)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

SCHEMA_VERSION = 1

INT_KEYS = {"steps", "seed", "clip_skip", "hires_steps"}
FLOAT_KEYS = {"cfg_scale", "denoising_strength", "hires_cfg_scale", "hires_upscale"}


def normalize_key(raw: str) -> str:
    return raw.strip().lower().replace(" ", "_")


def coerce(key: str, value: str) -> Any:
    v = value.strip()
    if key in INT_KEYS:
        try:
            return int(v)
        except ValueError:
            return v
    if key in FLOAT_KEYS:
        try:
            return float(v)
        except ValueError:
            return v
    return v


def parse_size(value: str) -> dict[str, int] | str:
    m = re.fullmatch(r"\s*(\d+)\s*x\s*(\d+)\s*", value)
    if not m:
        return value
    return {"width": int(m.group(1)), "height": int(m.group(2))}


def parse_a1111_parameters(text: str) -> dict[str, Any]:
    """Parse the A1111 / Forge `parameters` chunk into a structured dict."""
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_format": "a1111-parameters",
        "prompt": "",
        "negative_prompt": None,
        "params": {},
        "raw_parameters": text,
    }

    # Locate the params line: last line that starts with "Steps:".
    lines = text.splitlines()
    params_idx = None
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].lstrip().startswith("Steps:"):
            params_idx = i
            break

    body = "\n".join(lines[:params_idx]) if params_idx is not None else text
    params_line = lines[params_idx] if params_idx is not None else ""

    # Split positive / negative on the first "Negative prompt:" marker.
    neg_marker = "Negative prompt:"
    if neg_marker in body:
        pos, _, neg = body.partition(neg_marker)
        out["prompt"] = pos.strip()
        out["negative_prompt"] = neg.strip() or None
    else:
        out["prompt"] = body.strip()

    if not params_line:
        return out

    # Split the params line on ", " — robust for typical A1111 output.
    raw_params: dict[str, str] = {}
    for chunk in params_line.split(", "):
        if ": " not in chunk:
            continue
        k, _, v = chunk.partition(": ")
        raw_params[normalize_key(k)] = v

    params: dict[str, Any] = {}
    hires: dict[str, Any] = {}

    for k, v in raw_params.items():
        if k == "size":
            params["size"] = parse_size(v)
            continue
        if k.startswith("hires_") or k.startswith("hires "):
            sub = k[len("hires_"):] if k.startswith("hires_") else k[len("hires "):]
            sub = normalize_key(sub)
            full_key = f"hires_{sub}"
            hires[sub] = coerce(full_key, v)
            continue
        params[k] = coerce(k, v)

    if hires:
        params["hires"] = hires

    out["params"] = params
    return out


def parse_comfyui(prompt_chunk: str | None, workflow_chunk: str | None) -> dict[str, Any] | None:
    """Pass ComfyUI JSON chunks through as-is, parsed for convenience."""
    prompt_obj = None
    workflow_obj = None
    if prompt_chunk:
        try:
            prompt_obj = json.loads(prompt_chunk)
        except json.JSONDecodeError:
            return None
    if workflow_chunk:
        try:
            workflow_obj = json.loads(workflow_chunk)
        except json.JSONDecodeError:
            workflow_obj = None
    if prompt_obj is None and workflow_obj is None:
        return None
    return {
        "schema_version": SCHEMA_VERSION,
        "source_format": "comfyui",
        "prompt_graph": prompt_obj,
        "workflow_graph": workflow_obj,
    }


def parse_exif_user_comment(img: Image.Image) -> dict[str, Any] | None:
    """Some pipelines stash A1111-style text in EXIF UserComment (tag 37510)."""
    exif = img.getexif()
    if not exif:
        return None
    raw = exif.get(37510)
    if not raw:
        return None
    text = raw.decode("utf-8", errors="ignore") if isinstance(raw, bytes) else str(raw)
    # UserComment is prefixed with an 8-byte charset marker (e.g. b"UNICODE\x00")
    text = re.sub(r"^[A-Z]{5,8}\x00+", "", text).strip()
    if not text:
        return None
    if "Steps:" in text or "Negative prompt:" in text:
        return parse_a1111_parameters(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_format": "exif-user-comment",
        "raw_text": text,
    }


def extract(image_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Return (structured metadata, raw_parameters_string_or_None)."""
    with Image.open(image_path) as img:
        text_chunks = dict(getattr(img, "text", {}) or {})
        info = img.info or {}

        # 1. AUTOMATIC1111 / Forge
        params_chunk = text_chunks.get("parameters") or info.get("parameters")
        if isinstance(params_chunk, str) and params_chunk.strip():
            return parse_a1111_parameters(params_chunk), params_chunk

        # 2. ComfyUI
        prompt_chunk = text_chunks.get("prompt") or info.get("prompt")
        workflow_chunk = text_chunks.get("workflow") or info.get("workflow")
        comfy = parse_comfyui(
            prompt_chunk if isinstance(prompt_chunk, str) else None,
            workflow_chunk if isinstance(workflow_chunk, str) else None,
        )
        if comfy is not None:
            return comfy, None

        # 3. EXIF UserComment fallback
        exif_meta = parse_exif_user_comment(img)
        if exif_meta is not None:
            raw = exif_meta.get("raw_parameters") if exif_meta.get("source_format") == "a1111-parameters" else None
            return exif_meta, raw

    return None, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", type=Path, help="Path to the image file")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Directory for sidecar files (default: alongside the image)")
    ap.add_argument("--no-txt", action="store_true",
                    help="Skip writing the raw .txt sidecar (A1111 interop)")
    args = ap.parse_args()

    if not args.image.is_file():
        print(f"Error: not a file: {args.image}", file=sys.stderr)
        return 2

    metadata, raw = extract(args.image)
    if metadata is None:
        print(f"No generation metadata found in {args.image}", file=sys.stderr)
        return 1

    out_dir = args.out_dir or args.image.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem

    json_path = out_dir / f"{stem}.json"
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {json_path}")

    if raw and not args.no_txt:
        txt_path = out_dir / f"{stem}.txt"
        txt_path.write_text(raw, encoding="utf-8")
        print(f"Wrote {txt_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
