"""Smoke test: generate one image using the configured model.

Bypasses FastAPI, auth, queue, credits and storage. Loads the model directly
via GeneratorService and saves the result next to this script.

Usage (from project root, with .venv active):
    python scripts/smoke_generate.py "a fluffy white cat sitting on a windowsill at sunset"
"""

import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path when invoked as `python scripts/smoke_generate.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scratchy.config import get_settings
from scratchy.services.generator import GeneratorService


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    log = logging.getLogger("smoke")

    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "a fluffy white cat sitting on a windowsill at sunset, "
        "soft cinematic lighting, photorealistic, highly detailed"
    )

    settings = get_settings()
    log.info("Model name=%s pipeline=%s device=%s",
             settings.model.name, settings.model.pipeline_type, settings.model.device)
    log.info("Local path: %s", settings.model.local_path)

    gen = GeneratorService(settings.model)

    t0 = time.time()
    gen.load_model()
    log.info("load_model() done in %.1fs", time.time() - t0)

    gpu = gen.get_gpu_info()
    log.info("GPU: %s", gpu)

    # RealVisXL V5.0 Lightning recommends 4-8 steps with CFG 1.0-2.0.
    result = gen.generate(
        prompt=prompt,
        negative_prompt="blurry, low quality, deformed, watermark",
        width=1024,
        height=1024,
        steps=6,
        guidance_scale=1.5,
        seed=42,
        output_format="png",
    )

    out_dir = Path(__file__).resolve().parent.parent / "scratchy_data" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"smoke_{result.seed}.png"
    out_path.write_bytes(result.image_bytes)

    log.info("Generated in %.2fs (seed=%d), saved to %s",
             result.generation_time, result.seed, out_path)
    if result.warnings:
        log.warning("Warnings: %s", result.warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
