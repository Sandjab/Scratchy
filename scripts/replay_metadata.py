"""Regenerate an image from a metadata JSON file (the format produced by
`extract_metadata.py`), provided the referenced model is locally available.

Reads the JSON, locates the checkpoint inside `scratchy_data/models/`, maps
the A1111 sampler + schedule to a diffusers scheduler, applies clip_skip,
runs txt2img (with chunked long-prompt encoding to bypass the 77-token
CLIP limit), optionally runs an img2img hires-fix pass, and writes a PNG
that re-embeds the original `parameters` chunk — so the output is paste-
compatible with A1111 / Forge / ComfyUI.

Usage:
    python scripts/replay_metadata.py girl.json
    python scripts/replay_metadata.py girl.json -o out.png
    python scripts/replay_metadata.py girl.json --model PATH   # override search
    python scripts/replay_metadata.py girl.json --seed 1234    # override seed
    python scripts/replay_metadata.py girl.json --no-hires     # skip hires-fix

Limitations:
    * The hires upscaler in metadata (e.g. `4x_NickelbackFS_72000_G`) is not
      loaded — PIL Lanczos is used to scale before the img2img refinement.
    * ComfyUI graph metadata cannot be replayed (refused with an error).
    * Sampler match is best-effort; unmapped samplers fall back to the
      pipeline's default scheduler with a warning.
    * Long-prompt chunking is implemented for SDXL / SD1.5; FLUX falls back
      to truncated encoding.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image
from PIL.PngImagePlugin import PngInfo

log = logging.getLogger("replay")

SINGLE_FILE_EXTS = {".safetensors", ".ckpt", ".pt", ".bin"}

# (sampler_lower, schedule_lower) -> (scheduler_class_name, kwargs)
# schedule_lower of "" matches when schedule_type is missing.
SCHEDULER_MAP: dict[tuple[str, str], tuple[str, dict[str, Any]]] = {
    ("dpm++ 2m", ""):              ("DPMSolverMultistepScheduler", {}),
    ("dpm++ 2m", "karras"):        ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    ("dpm++ 2m sde", ""):          ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    ("dpm++ 2m sde", "karras"):    ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True}),
    ("dpm++ sde", ""):             ("DPMSolverSinglestepScheduler", {}),
    ("dpm++ sde", "karras"):       ("DPMSolverSinglestepScheduler", {"use_karras_sigmas": True}),
    ("euler", ""):                 ("EulerDiscreteScheduler", {}),
    ("euler", "karras"):           ("EulerDiscreteScheduler", {"use_karras_sigmas": True}),
    ("euler a", ""):               ("EulerAncestralDiscreteScheduler", {}),
    ("euler ancestral", ""):       ("EulerAncestralDiscreteScheduler", {}),
    ("ddim", ""):                  ("DDIMScheduler", {}),
    ("lms", ""):                   ("LMSDiscreteScheduler", {}),
    ("lms", "karras"):             ("LMSDiscreteScheduler", {"use_karras_sigmas": True}),
    ("heun", ""):                  ("HeunDiscreteScheduler", {}),
    ("dpm2", ""):                  ("KDPM2DiscreteScheduler", {}),
    ("dpm2", "karras"):            ("KDPM2DiscreteScheduler", {"use_karras_sigmas": True}),
    ("dpm2 a", ""):                ("KDPM2AncestralDiscreteScheduler", {}),
    ("unipc", ""):                 ("UniPCMultistepScheduler", {}),
}


def find_model(model_name_hint: str, models_root: Path) -> Path | None:
    """Search `models_root` for a checkpoint whose name contains the hint.

    Hint matching is case-insensitive and ignores common separators. Returns
    the path on a unique match, None on no match. Raises on ambiguity.
    """
    if not models_root.exists():
        return None

    norm_hint = _normalize(model_name_hint)
    if not norm_hint:
        return None

    candidates: list[Path] = []
    for path in models_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in SINGLE_FILE_EXTS:
            haystacks = [_normalize(path.stem)]
            sidecar = path.parent / "metadata.json"
            if sidecar.exists():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    for k in ("model_name", "version_name", "filename"):
                        v = meta.get(k)
                        if isinstance(v, str):
                            haystacks.append(_normalize(v))
                except (OSError, json.JSONDecodeError):
                    pass
            if any(_hint_matches(norm_hint, h) for h in haystacks):
                candidates.append(path)

    if not candidates:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            f"Ambiguous model match for {model_name_hint!r}: "
            + ", ".join(str(p.relative_to(models_root)) for p in candidates)
        )
    return candidates[0]


def list_available_models(models_root: Path) -> list[Path]:
    if not models_root.exists():
        return []
    return sorted(
        p for p in models_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SINGLE_FILE_EXTS
    )


def _normalize(s: str) -> str:
    return "".join(c.lower() for c in s if c.isalnum())


def _hint_matches(norm_hint: str, norm_haystack: str) -> bool:
    """Bidirectional substring match — A1111 model names often carry quant
    suffixes (`_FP32`, `_FP16`) absent from the on-disk filename, so we
    accept hint⊂haystack and haystack⊂hint."""
    if not norm_hint or not norm_haystack:
        return False
    return norm_hint in norm_haystack or norm_haystack in norm_hint


def detect_pipeline_type(model_path: Path, model_meta: dict[str, Any]) -> str:
    """Detect SDXL / SD1.5 / FLUX from sidecar metadata or filename."""
    sidecar = model_path.parent / "metadata.json"
    if sidecar.exists():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
            base = (meta.get("base_model") or "").lower()
            if "flux" in base:
                return "flux"
            if "xl" in base or "pony" in base:
                return "sdxl"
            if "1.5" in base or "sd 1" in base:
                return "sd15"
        except (OSError, json.JSONDecodeError):
            pass

    name = model_path.stem.lower() + " " + (model_meta.get("model") or "").lower()
    if "flux" in name:
        return "flux"
    if "xl" in name or "pony" in name:
        return "sdxl"
    if "sd15" in name or "sd_1" in name or "1.5" in name:
        return "sd15"
    log.warning("Could not detect pipeline type, defaulting to SDXL")
    return "sdxl"


def build_scheduler(pipe, sampler: str | None, schedule_type: str | None) -> list[str]:
    """Replace pipe.scheduler with one matching the A1111 sampler+schedule.

    Returns a list of warning strings.
    """
    warnings: list[str] = []
    if not sampler:
        return ["sampler not specified in metadata; using pipeline default"]

    key = (sampler.lower().strip(), (schedule_type or "").lower().strip())
    entry = SCHEDULER_MAP.get(key)
    if entry is None and key[1]:
        # Try without schedule
        entry = SCHEDULER_MAP.get((key[0], ""))
        if entry is not None:
            warnings.append(f"schedule_type={schedule_type!r} unmapped, falling back to plain {sampler}")

    if entry is None:
        return [f"sampler {sampler!r} (schedule={schedule_type!r}) unmapped; using pipeline default"]

    cls_name, kwargs = entry
    import diffusers
    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        return [f"diffusers has no scheduler {cls_name!r}; using pipeline default"]

    pipe.scheduler = cls.from_config(pipe.scheduler.config, **kwargs)
    log.info("Scheduler set to %s%s", cls_name, f" (kwargs={kwargs})" if kwargs else "")
    return warnings


def _chunk_token_ids(input_ids: list[int], bos: int, eos: int, pad: int, max_len: int) -> list[list[int]]:
    """Split a flat sequence of input ids (already without BOS/EOS) into
    fixed-size windows, each padded with BOS/EOS/PAD up to `max_len`."""
    chunk_len = max_len - 2  # reserve room for BOS + EOS
    if not input_ids:
        windows = [[]]
    else:
        windows = [input_ids[i:i + chunk_len] for i in range(0, len(input_ids), chunk_len)]
    out = []
    for w in windows:
        full = [bos] + w + [eos]
        full += [pad] * (max_len - len(full))
        out.append(full)
    return out


def _strip_special(ids: list[int], bos: int, eos: int) -> list[int]:
    if ids and ids[0] == bos:
        ids = ids[1:]
    while ids and ids[-1] == eos:
        ids = ids[:-1]
    return ids


def encode_long_prompt_sdxl(pipe, prompt: str, negative: str | None,
                             clip_skip: int | None, device: str, dtype):
    """Encode a (possibly very long) prompt for SDXL by chunking into 75-token
    windows. Returns positive/negative prompt embeds and pooled embeds in the
    shapes expected by `StableDiffusionXLPipeline.__call__`."""
    import torch
    tokenizers = [pipe.tokenizer, pipe.tokenizer_2]
    text_encoders = [pipe.text_encoder, pipe.text_encoder_2]
    # A1111 clip_skip: 1 = last layer, 2 = penultimate. SDXL diffusers default
    # is penultimate (-2). Map A1111 N -> hidden_states[-N] capped at -2.
    skip_idx = -max(2, int(clip_skip)) if clip_skip else -2

    def encode(text: str):
        per_encoder_embeds = []
        pooled = None
        for tok, enc in zip(tokenizers, text_encoders):
            max_len = tok.model_max_length  # 77
            bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
            eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
            pad = tok.pad_token_id if tok.pad_token_id is not None else eos

            ids = tok(text, padding=False, truncation=False,
                      add_special_tokens=True, return_tensors=None)["input_ids"]
            ids = _strip_special(ids, bos, eos)
            windows = _chunk_token_ids(ids, bos, eos, pad, max_len)
            window_embeds = []
            window_pooled = None
            for w in windows:
                t = torch.tensor([w], device=device)
                out = enc(t, output_hidden_states=True)
                # SDXL uses non-final hidden state from each encoder.
                hs = out.hidden_states[skip_idx] if abs(skip_idx) <= len(out.hidden_states) else out.hidden_states[-2]
                window_embeds.append(hs)
                if window_pooled is None and hasattr(out, "text_embeds"):
                    window_pooled = out.text_embeds  # only first chunk's pooled
            per_encoder_embeds.append(torch.cat(window_embeds, dim=1))
            if pooled is None and window_pooled is not None:
                pooled = window_pooled
        # Concat the per-encoder embeddings along the channel axis (SDXL).
        prompt_embeds = torch.cat(per_encoder_embeds, dim=-1).to(dtype=dtype)
        return prompt_embeds, (pooled.to(dtype=dtype) if pooled is not None else None)

    pos_embeds, pos_pooled = encode(prompt)
    neg_text = negative or ""
    neg_embeds, neg_pooled = encode(neg_text)

    # Pad the shorter sequence so positive and negative align in length.
    if pos_embeds.shape[1] != neg_embeds.shape[1]:
        target = max(pos_embeds.shape[1], neg_embeds.shape[1])

        def pad_to(emb, n):
            if emb.shape[1] >= n:
                return emb[:, :n]
            extra = n - emb.shape[1]
            tail = emb[:, -1:].expand(-1, extra, -1)
            return torch.cat([emb, tail], dim=1)

        pos_embeds = pad_to(pos_embeds, target)
        neg_embeds = pad_to(neg_embeds, target)

    return pos_embeds, pos_pooled, neg_embeds, neg_pooled


def encode_long_prompt_sd15(pipe, prompt: str, negative: str | None,
                             clip_skip: int | None, device: str, dtype):
    """Same idea for SD 1.5 (single CLIP-L encoder, no pooled embeds)."""
    import torch
    tok = pipe.tokenizer
    enc = pipe.text_encoder
    max_len = tok.model_max_length
    bos = tok.bos_token_id if tok.bos_token_id is not None else tok.cls_token_id
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.sep_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else eos
    skip_idx = -max(1, int(clip_skip)) if clip_skip else -1

    def encode(text: str):
        ids = tok(text, padding=False, truncation=False,
                  add_special_tokens=True, return_tensors=None)["input_ids"]
        ids = _strip_special(ids, bos, eos)
        windows = _chunk_token_ids(ids, bos, eos, pad, max_len)
        embeds = []
        for w in windows:
            t = torch.tensor([w], device=device)
            out = enc(t, output_hidden_states=True)
            hs = out.hidden_states[skip_idx] if abs(skip_idx) <= len(out.hidden_states) else out.hidden_states[-1]
            # Apply final layer norm if the encoder has one (CLIPTextModel does).
            final_ln = getattr(getattr(enc, "text_model", enc), "final_layer_norm", None)
            if skip_idx != -1 and final_ln is not None:
                hs = final_ln(hs)
            embeds.append(hs)
        return torch.cat(embeds, dim=1).to(dtype=dtype)

    pos = encode(prompt)
    neg = encode(negative or "")
    if pos.shape[1] != neg.shape[1]:
        target = max(pos.shape[1], neg.shape[1])

        def pad_to(emb, n):
            if emb.shape[1] >= n:
                return emb[:, :n]
            extra = n - emb.shape[1]
            tail = emb[:, -1:].expand(-1, extra, -1)
            return torch.cat([emb, tail], dim=1)

        pos = pad_to(pos, target)
        neg = pad_to(neg, target)
    return pos, neg


def needs_long_prompt(pipe, prompt: str, negative: str | None) -> bool:
    """Return True if either prompt exceeds the tokenizer's 77-token limit."""
    tok = getattr(pipe, "tokenizer", None)
    if tok is None:
        return False
    max_len = tok.model_max_length
    for text in (prompt, negative or ""):
        if not text:
            continue
        ids = tok(text, padding=False, truncation=False, return_tensors=None)["input_ids"]
        if len(ids) > max_len:
            return True
    return False


def build_img2img_pipeline(base_pipe, pipeline_type: str):
    """Construct an img2img pipeline that shares weights with `base_pipe`."""
    if pipeline_type == "sdxl":
        from diffusers import StableDiffusionXLImg2ImgPipeline
        return StableDiffusionXLImg2ImgPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            text_encoder_2=base_pipe.text_encoder_2,
            tokenizer=base_pipe.tokenizer,
            tokenizer_2=base_pipe.tokenizer_2,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
        )
    if pipeline_type == "sd15":
        from diffusers import StableDiffusionImg2ImgPipeline
        return StableDiffusionImg2ImgPipeline(
            vae=base_pipe.vae,
            text_encoder=base_pipe.text_encoder,
            tokenizer=base_pipe.tokenizer,
            unet=base_pipe.unet,
            scheduler=base_pipe.scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
    raise ValueError(f"hires-fix not supported for pipeline_type={pipeline_type!r}")


def run_hires_pass(base_pipe, pipeline_type: str, base_image: Image.Image,
                   prompt: str, negative: str | None, hires: dict[str, Any],
                   base_size: tuple[int, int], cfg_default: float,
                   clip_skip: int | None, device: str, dtype) -> Image.Image:
    """Replay an A1111-style hires-fix: PIL Lanczos upscale + img2img refinement."""
    import torch
    upscale = float(hires.get("upscale") or 1.5)
    h_steps = int(hires.get("steps") or 12)
    h_cfg = float(hires.get("cfg_scale") or cfg_default)
    h_denoise = float(hires.get("denoising_strength") or hires.get("denoise") or 0.35)

    base_w, base_h = base_size
    target_w = int(round(base_w * upscale / 8.0)) * 8
    target_h = int(round(base_h * upscale / 8.0)) * 8
    log.info("Hires-fix: %dx%d -> %dx%d (Lanczos upscale, denoise=%.2f, %d steps, cfg=%.2f)",
             base_w, base_h, target_w, target_h, h_denoise, h_steps, h_cfg)
    upscaler_name = hires.get("upscaler")
    if upscaler_name and upscaler_name.lower() not in {"latent", "none", ""}:
        log.warning("Hires upscaler %r not loaded; using PIL Lanczos as substitute",
                    upscaler_name)

    enlarged = base_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

    img2img = build_img2img_pipeline(base_pipe, pipeline_type)

    call_kwargs: dict[str, Any] = {
        "image": enlarged,
        "num_inference_steps": h_steps,
        "guidance_scale": h_cfg,
        "strength": h_denoise,
    }

    if pipeline_type == "sdxl" and needs_long_prompt(base_pipe, prompt, negative):
        pos, pos_p, neg, neg_p = encode_long_prompt_sdxl(
            base_pipe, prompt, negative, clip_skip, device, dtype)
        call_kwargs.update({
            "prompt_embeds": pos, "negative_prompt_embeds": neg,
            "pooled_prompt_embeds": pos_p, "negative_pooled_prompt_embeds": neg_p,
        })
    elif pipeline_type == "sd15" and needs_long_prompt(base_pipe, prompt, negative):
        pos, neg = encode_long_prompt_sd15(
            base_pipe, prompt, negative, clip_skip, device, dtype)
        call_kwargs.update({"prompt_embeds": pos, "negative_prompt_embeds": neg})
    else:
        call_kwargs["prompt"] = prompt
        if negative:
            call_kwargs["negative_prompt"] = negative
        if isinstance(clip_skip, int) and clip_skip > 0:
            call_kwargs["clip_skip"] = clip_skip

    with torch.inference_mode():
        result = img2img(**call_kwargs)
    return result.images[0]


def load_pipeline(model_path: Path, pipeline_type: str, dtype):
    log.info("Loading %s pipeline from %s", pipeline_type, model_path.name)
    if pipeline_type == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        return StableDiffusionXLPipeline.from_single_file(str(model_path), torch_dtype=dtype)
    if pipeline_type == "sd15":
        from diffusers import StableDiffusionPipeline
        return StableDiffusionPipeline.from_single_file(str(model_path), torch_dtype=dtype)
    if pipeline_type == "flux":
        from diffusers import FluxPipeline
        return FluxPipeline.from_single_file(str(model_path), torch_dtype=dtype)
    from diffusers import DiffusionPipeline
    return DiffusionPipeline.from_single_file(str(model_path), torch_dtype=dtype)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path, help="Path to a metadata JSON (from extract_metadata.py)")
    ap.add_argument("-o", "--output", type=Path, default=None,
                    help="Output PNG path (default: <metadata-stem>_replay.png next to JSON)")
    ap.add_argument("--model", type=Path, default=None,
                    help="Override model path (skip the local search)")
    ap.add_argument("--models-dir", type=Path,
                    default=REPO_ROOT / "scratchy_data" / "models",
                    help="Where to search for checkpoints")
    ap.add_argument("--seed", type=int, default=None, help="Override seed")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--no-hires", action="store_true",
                    help="Skip the hires-fix pass even if metadata has one")
    ap.add_argument("--no-long-prompt", action="store_true",
                    help="Disable chunked long-prompt encoding (truncate at 77 tokens)")
    args = ap.parse_args()

    if not args.metadata.is_file():
        log.error("metadata file not found: %s", args.metadata)
        return 2

    meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    src_format = meta.get("source_format")
    if src_format != "a1111-parameters":
        log.error("source_format=%r is not replayable (only 'a1111-parameters' is supported)",
                  src_format)
        return 2

    params = meta.get("params", {}) or {}
    prompt = (meta.get("prompt") or "").strip()
    negative = meta.get("negative_prompt")
    if not prompt:
        log.error("metadata has no prompt")
        return 2

    # Resolve model
    if args.model:
        model_path = args.model
        if not model_path.is_file():
            log.error("--model path is not a file: %s", model_path)
            return 2
    else:
        model_hint = params.get("model") or ""
        if not model_hint:
            log.error("metadata has no params.model and --model was not provided")
            return 2
        try:
            found = find_model(model_hint, args.models_dir)
        except RuntimeError as e:
            log.error("%s", e)
            return 2
        if found is None:
            available = list_available_models(args.models_dir)
            log.error("Model %r not available locally under %s", model_hint, args.models_dir)
            if available:
                log.error("Available checkpoints:")
                for p in available:
                    log.error("  - %s", p.relative_to(args.models_dir))
            else:
                log.error("(no checkpoints found)")
            log.error("Download it with: scratchy-models download <civitai_url>")
            log.error("Or pass --model PATH to use any local checkpoint.")
            return 1
        model_path = found
        log.info("Resolved model %r -> %s", model_hint, model_path)

    pipeline_type = detect_pipeline_type(model_path, params)
    log.info("Pipeline type: %s", pipeline_type)

    # Heavy imports only after validation
    import torch
    dtype = torch.float16 if args.device == "cuda" else torch.float32

    pipe = load_pipeline(model_path, pipeline_type, dtype)
    pipe = pipe.to(args.device)
    if hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing()
    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()

    warnings = build_scheduler(pipe, params.get("sampler"), params.get("schedule_type"))

    # Resolve generation params
    seed = args.seed if args.seed is not None else params.get("seed")
    if seed is None:
        seed = torch.randint(0, 2**32 - 1, (1,)).item()
    steps = int(params.get("steps") or 30)
    cfg = float(params.get("cfg_scale") or 7.0)
    size = params.get("size") or {}
    width = int(size.get("width") or 1024)
    height = int(size.get("height") or 1024)
    clip_skip = params.get("clip_skip")

    log.info("Generating %dx%d, steps=%d, cfg=%.2f, seed=%d, clip_skip=%s",
             width, height, steps, cfg, seed, clip_skip)

    gen = torch.Generator(device=args.device).manual_seed(int(seed))
    call_kwargs: dict[str, Any] = {
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": cfg,
        "generator": gen,
    }

    long_prompt = (not args.no_long_prompt) and needs_long_prompt(pipe, prompt, negative)
    if long_prompt and pipeline_type == "sdxl":
        log.info("Prompt exceeds CLIP 77-token limit; using chunked SDXL encoding")
        pos, pos_p, neg, neg_p = encode_long_prompt_sdxl(
            pipe, prompt, negative, clip_skip if isinstance(clip_skip, int) else None,
            args.device, dtype)
        call_kwargs.update({
            "prompt_embeds": pos, "negative_prompt_embeds": neg,
            "pooled_prompt_embeds": pos_p, "negative_pooled_prompt_embeds": neg_p,
        })
    elif long_prompt and pipeline_type == "sd15":
        log.info("Prompt exceeds CLIP 77-token limit; using chunked SD1.5 encoding")
        pos, neg = encode_long_prompt_sd15(
            pipe, prompt, negative, clip_skip if isinstance(clip_skip, int) else None,
            args.device, dtype)
        call_kwargs.update({"prompt_embeds": pos, "negative_prompt_embeds": neg})
    else:
        if long_prompt:
            warnings.append(f"long-prompt chunking not implemented for {pipeline_type}; "
                            f"prompt will be truncated at 77 tokens")
        call_kwargs["prompt"] = prompt
        if negative:
            call_kwargs["negative_prompt"] = negative
        if isinstance(clip_skip, int) and clip_skip > 0:
            call_kwargs["clip_skip"] = clip_skip

    t0 = time.time()
    with torch.inference_mode():
        result = pipe(**call_kwargs)
    elapsed = time.time() - t0
    log.info("Base generated in %.2fs", elapsed)

    image: Image.Image = result.images[0]

    hires = params.get("hires") if isinstance(params.get("hires"), dict) else None
    if hires and not args.no_hires:
        if pipeline_type in {"sdxl", "sd15"}:
            t1 = time.time()
            image = run_hires_pass(
                pipe, pipeline_type, image, prompt, negative, hires,
                base_size=(width, height),
                cfg_default=cfg,
                clip_skip=clip_skip if isinstance(clip_skip, int) else None,
                device=args.device, dtype=dtype,
            )
            log.info("Hires-fix done in %.2fs", time.time() - t1)
        else:
            warnings.append(f"hires-fix block ignored: not supported for {pipeline_type}")
    elif hires and args.no_hires:
        warnings.append("hires-fix block found but skipped (--no-hires)")

    out_path = args.output
    if out_path is None:
        out_path = args.metadata.with_name(args.metadata.stem + "_replay.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Re-embed the original `parameters` chunk so the output is paste-
    # compatible with A1111 / Forge / ComfyUI.
    pnginfo = PngInfo()
    raw = meta.get("raw_parameters")
    if isinstance(raw, str) and raw:
        pnginfo.add_text("parameters", raw)
    pnginfo.add_text("scratchy_replay_seed", str(seed))
    pnginfo.add_text("scratchy_replay_source", str(args.metadata.name))
    image.save(out_path, format="PNG", pnginfo=pnginfo)

    log.info("Saved %s", out_path)
    for w in warnings:
        log.warning("%s", w)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
