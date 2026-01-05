"""
Serveur API minimaliste de génération d'images
Supporte : FLUX.1-schnell, FLUX.1-dev, Z-Image-Turbo, SDXL

Usage:
    pip install fastapi uvicorn diffusers torch accelerate
    # Pour Z-Image: pip install git+https://github.com/huggingface/diffusers
    
    python server.py
    
API:
    POST /generate
    {
        "prompt": "a cat in space",
        "width": 1024,
        "height": 1024,
        "steps": 8,
        "seed": null
    }
"""

import io
import base64
import time
import logging
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field

# ============================================================================
# CONFIGURATION - Modifier ici selon tes besoins
# ============================================================================

MODEL_CONFIG = {
    # Choisis un modèle en décommentant la ligne correspondante:
    
    # FLUX.1-schnell - Ultra rapide (4-8 steps), Apache 2.0 license
    "name": "black-forest-labs/FLUX.1-schnell",
    "pipeline": "flux",
    "default_steps": 4,
    "guidance_scale": 0.0,  # FLUX schnell n'utilise pas de guidance
    
    # FLUX.1-dev - Meilleure qualité (20-30 steps), non-commercial
    # "name": "black-forest-labs/FLUX.1-dev",
    # "pipeline": "flux",
    # "default_steps": 28,
    # "guidance_scale": 3.5,
    
    # Z-Image-Turbo - 8 steps, excellent rendu texte, Apache 2.0
    # "name": "Tongyi-MAI/Z-Image-Turbo",
    # "pipeline": "zimage",
    # "default_steps": 8,
    # "guidance_scale": 1.0,
    
    # SDXL - Classique, bon écosystème LoRA
    # "name": "stabilityai/stable-diffusion-xl-base-1.0",
    # "pipeline": "sdxl",
    # "default_steps": 30,
    # "guidance_scale": 7.5,
}

# Paramètres serveur
HOST = "0.0.0.0"
PORT = 8080
DEVICE = "cuda"  # ou "cpu", "mps" pour Mac
DTYPE = torch.float16  # torch.bfloat16 pour Z-Image si supporté

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# MODÈLES PYDANTIC
# ============================================================================

class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Le prompt de génération")
    negative_prompt: Optional[str] = Field(None, description="Prompt négatif (optionnel)")
    width: int = Field(1024, ge=256, le=2048, description="Largeur de l'image")
    height: int = Field(1024, ge=256, le=2048, description="Hauteur de l'image")
    steps: Optional[int] = Field(None, ge=1, le=100, description="Nombre de steps")
    guidance_scale: Optional[float] = Field(None, ge=0.0, le=20.0, description="CFG scale")
    seed: Optional[int] = Field(None, description="Seed pour reproductibilité")
    output_format: str = Field("png", description="Format: png, jpeg, webp")


class GenerateResponse(BaseModel):
    image_base64: str
    seed: int
    generation_time_ms: int
    model: str


class HealthResponse(BaseModel):
    status: str
    model: str
    device: str


# ============================================================================
# CHARGEMENT DU PIPELINE
# ============================================================================

pipe = None

def load_pipeline():
    """Charge le pipeline de diffusion selon la configuration."""
    global pipe
    
    model_name = MODEL_CONFIG["name"]
    pipeline_type = MODEL_CONFIG["pipeline"]
    
    logger.info(f"Chargement du modèle: {model_name}")
    start = time.time()
    
    if pipeline_type == "flux":
        from diffusers import FluxPipeline
        pipe = FluxPipeline.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
        )
    elif pipeline_type == "zimage":
        from diffusers import ZImagePipeline
        pipe = ZImagePipeline.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # Z-Image préfère bfloat16
        )
    elif pipeline_type == "sdxl":
        from diffusers import StableDiffusionXLPipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            model_name,
            torch_dtype=DTYPE,
            use_safetensors=True,
            variant="fp16",
        )
    else:
        raise ValueError(f"Pipeline inconnu: {pipeline_type}")
    
    pipe.to(DEVICE)
    
    # Optimisations mémoire
    if hasattr(pipe, 'enable_attention_slicing'):
        pipe.enable_attention_slicing()
    
    # VAE slicing pour grandes images
    if hasattr(pipe, 'enable_vae_slicing'):
        pipe.enable_vae_slicing()
    
    elapsed = time.time() - start
    logger.info(f"Modèle chargé en {elapsed:.1f}s")
    
    return pipe


# ============================================================================
# APPLICATION FASTAPI
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage."""
    load_pipeline()
    yield
    # Cleanup si nécessaire


app = FastAPI(
    title="Image Generation API",
    description="API minimaliste de génération d'images avec Diffusers",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS pour app mobile
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health():
    """Vérifie que le serveur et le modèle sont prêts."""
    return HealthResponse(
        status="ok" if pipe is not None else "loading",
        model=MODEL_CONFIG["name"],
        device=DEVICE,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Génère une image à partir d'un prompt."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    start = time.time()
    
    # Seed
    seed = request.seed if request.seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    # Paramètres avec fallback sur config
    steps = request.steps or MODEL_CONFIG["default_steps"]
    guidance = request.guidance_scale if request.guidance_scale is not None else MODEL_CONFIG["guidance_scale"]
    
    # Construction des arguments
    gen_kwargs = {
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": steps,
        "generator": generator,
    }
    
    # Guidance scale (certains modèles ne l'utilisent pas)
    if guidance > 0:
        gen_kwargs["guidance_scale"] = guidance
    
    # Negative prompt (si supporté)
    if request.negative_prompt and MODEL_CONFIG["pipeline"] in ["sdxl"]:
        gen_kwargs["negative_prompt"] = request.negative_prompt
    
    try:
        with torch.inference_mode():
            result = pipe(**gen_kwargs)
            image = result.images[0]
    except Exception as e:
        logger.error(f"Erreur génération: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    # Conversion en bytes
    buffer = io.BytesIO()
    format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
    img_format = format_map.get(request.output_format, "PNG")
    image.save(buffer, format=img_format, quality=95 if img_format == "JPEG" else None)
    buffer.seek(0)
    
    elapsed_ms = int((time.time() - start) * 1000)
    logger.info(f"Image générée en {elapsed_ms}ms (seed={seed})")
    
    return GenerateResponse(
        image_base64=base64.b64encode(buffer.getvalue()).decode(),
        seed=seed,
        generation_time_ms=elapsed_ms,
        model=MODEL_CONFIG["name"],
    )


@app.post("/generate/raw")
async def generate_raw(request: GenerateRequest):
    """Génère une image et retourne directement les bytes (pas de base64)."""
    if pipe is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    seed = request.seed if request.seed is not None else torch.randint(0, 2**32 - 1, (1,)).item()
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    steps = request.steps or MODEL_CONFIG["default_steps"]
    guidance = request.guidance_scale if request.guidance_scale is not None else MODEL_CONFIG["guidance_scale"]
    
    gen_kwargs = {
        "prompt": request.prompt,
        "width": request.width,
        "height": request.height,
        "num_inference_steps": steps,
        "generator": generator,
    }
    
    if guidance > 0:
        gen_kwargs["guidance_scale"] = guidance
    
    try:
        with torch.inference_mode():
            result = pipe(**gen_kwargs)
            image = result.images[0]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
    buffer = io.BytesIO()
    content_type = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }
    format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
    img_format = format_map.get(request.output_format, "PNG")
    image.save(buffer, format=img_format)
    buffer.seek(0)
    
    return Response(
        content=buffer.getvalue(),
        media_type=content_type.get(request.output_format, "image/png"),
        headers={"X-Seed": str(seed)},
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
