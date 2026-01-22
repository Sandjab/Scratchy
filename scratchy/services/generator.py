"""Image generation service."""

import io
import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass

from PIL import Image

from scratchy.config import ModelSettings, get_settings

# Lazy import torch to avoid loading ML dependencies if not needed
if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# File extensions indicating single-file checkpoints
SINGLE_FILE_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin"}


@dataclass
class GenerationResult:
    """Result of an image generation."""
    image: Image.Image
    image_bytes: bytes
    seed: int
    generation_time: float
    output_format: str
    warnings: list[str]


class GeneratorService:
    """Service for generating images using diffusion models."""

    def __init__(self, settings: ModelSettings):
        """
        Initialize the generator service.

        Args:
            settings: Model configuration settings
        """
        self._settings = settings
        self._pipe = None
        self._device = settings.device
        self._model_name = settings.model_id

    @property
    def is_loaded(self) -> bool:
        """Check if the model is loaded."""
        return self._pipe is not None

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._settings.name

    def load_model(self) -> None:
        """Load the diffusion model."""
        import torch  # Lazy import

        logger.info(f"Loading model: {self._model_name}")
        start = time.time()

        # Determine dtype based on quantization and device
        if self._settings.quantization == "8bit":
            dtype = torch.float16
            load_in_8bit = True
            load_in_4bit = False
        elif self._settings.quantization == "4bit":
            dtype = torch.float16
            load_in_8bit = False
            load_in_4bit = True
        else:
            dtype = torch.float16 if self._device == "cuda" else torch.float32
            load_in_8bit = False
            load_in_4bit = False

        # Use bfloat16 for Z-Image if available
        if self._settings.name == "z-turbo" and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        # Resolve model path based on source (local > civitai > url > huggingface)
        model_path = self._resolve_model_path()

        # Detect if it's a single file checkpoint vs diffusers format
        is_single_file = self._is_single_file(model_path)

        # Load appropriate pipeline
        pipeline_type = self._get_pipeline_type(model_path, is_single_file)

        if is_single_file:
            self._load_from_single_file(model_path, pipeline_type, dtype)
        else:
            self._load_from_pretrained(model_path, pipeline_type, dtype)

        # Move to device
        self._pipe.to(self._device)

        # Apply memory optimizations
        if hasattr(self._pipe, 'enable_attention_slicing'):
            self._pipe.enable_attention_slicing()

        if hasattr(self._pipe, 'vae') and hasattr(self._pipe.vae, 'enable_slicing'):
            self._pipe.vae.enable_slicing()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def _resolve_model_path(self) -> str:
        """
        Resolve the model path based on configured source.

        Priority: local_path > civitai > download_url > huggingface

        Returns:
            Path string (local path or HuggingFace model ID)
        """
        if self._settings.name != "custom":
            return self._settings.model_id

        # Custom model - check sources in priority order
        if self._settings.local_path:
            path = Path(self._settings.local_path)
            if not path.exists():
                raise FileNotFoundError(f"Local model path not found: {path}")
            logger.info(f"Using local model: {path}")
            return str(path)

        if self._settings.civitai_model_id:
            return self._get_or_download_civitai_model()

        if self._settings.download_url:
            return self._get_or_download_from_url()

        raise ValueError(
            "Custom model requires local_path, civitai_model_id, or download_url"
        )

    def _get_or_download_civitai_model(self) -> str:
        """Download from CivitAI if not cached, return path."""
        # Import directly to avoid __init__.py issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "downloader",
            Path(__file__).parent / "downloader.py"
        )
        downloader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader_module)
        ModelDownloader = downloader_module.ModelDownloader

        settings = get_settings()
        downloader = ModelDownloader(settings.storage.models_dir)

        # Check if already downloaded
        existing = downloader.get_model_path(self._settings.civitai_model_id)
        if existing:
            logger.info(f"Using cached CivitAI model: {existing}")
            return str(existing)

        # Download with progress logging
        def progress_callback(progress):
            pct = (progress.downloaded_bytes / progress.total_bytes * 100
                   if progress.total_bytes else 0)
            speed_mb = progress.speed_bytes_per_sec / (1024 * 1024)
            logger.info(f"Downloading: {pct:.1f}% ({speed_mb:.1f} MB/s)")

        model_path = downloader.download_from_civitai(
            model_id=self._settings.civitai_model_id,
            version_id=self._settings.civitai_version_id,
            progress_callback=progress_callback,
        )

        return str(model_path)

    def _get_or_download_from_url(self) -> str:
        """Download from URL if not cached, return path."""
        # Import directly to avoid __init__.py issues
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "downloader",
            Path(__file__).parent / "downloader.py"
        )
        downloader_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(downloader_module)
        ModelDownloader = downloader_module.ModelDownloader

        settings = get_settings()
        downloader = ModelDownloader(settings.storage.models_dir)

        # Check if URL is already downloaded (by checking URL hash directory)
        import hashlib
        url_hash = hashlib.sha256(self._settings.download_url.encode()).hexdigest()[:12]
        url_dir = settings.storage.models_dir / "url" / url_hash

        if url_dir.exists():
            for f in url_dir.iterdir():
                if f.suffix in SINGLE_FILE_EXTENSIONS:
                    logger.info(f"Using cached URL model: {f}")
                    return str(f)

        # Download with progress logging
        def progress_callback(progress):
            pct = (progress.downloaded_bytes / progress.total_bytes * 100
                   if progress.total_bytes else 0)
            speed_mb = progress.speed_bytes_per_sec / (1024 * 1024)
            logger.info(f"Downloading: {pct:.1f}% ({speed_mb:.1f} MB/s)")

        model_path = downloader.download_from_url(
            url=self._settings.download_url,
            progress_callback=progress_callback,
        )

        return str(model_path)

    def _is_single_file(self, model_path: str) -> bool:
        """Check if model path is a single file checkpoint."""
        path = Path(model_path)

        # If it's a file with checkpoint extension
        if path.is_file() and path.suffix in SINGLE_FILE_EXTENSIONS:
            return True

        # If it's a HuggingFace model ID (contains slash, not a path)
        if "/" in model_path and not path.exists():
            return False

        # If it's a directory, check for diffusers structure
        if path.is_dir():
            # Diffusers format has model_index.json
            if (path / "model_index.json").exists():
                return False
            # Or at least unet folder
            if (path / "unet").is_dir():
                return False
            # Directory with only checkpoint files
            checkpoint_files = [f for f in path.iterdir() if f.suffix in SINGLE_FILE_EXTENSIONS]
            if checkpoint_files:
                return True

        return False

    def _load_from_single_file(self, model_path: str, pipeline_type: str, dtype) -> None:
        """Load model from a single file checkpoint."""
        logger.info(f"Loading single file checkpoint: {model_path}")

        if pipeline_type == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            self._pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
        elif pipeline_type == "sd15":
            from diffusers import StableDiffusionPipeline
            self._pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
        elif pipeline_type == "flux":
            from diffusers import FluxPipeline
            self._pipe = FluxPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )
        else:
            # Try auto-detection with DiffusionPipeline
            from diffusers import DiffusionPipeline
            self._pipe = DiffusionPipeline.from_single_file(
                model_path,
                torch_dtype=dtype,
            )

    def _load_from_pretrained(self, model_path: str, pipeline_type: str, dtype) -> None:
        """Load model from pretrained (HuggingFace or local diffusers format)."""
        if pipeline_type == "flux":
            from diffusers import FluxPipeline
            self._pipe = FluxPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
        elif pipeline_type == "zimage":
            from diffusers import DiffusionPipeline
            self._pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
        elif pipeline_type == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if dtype.is_floating_point else None,
            )
        elif pipeline_type == "sd15":
            from diffusers import StableDiffusionPipeline
            self._pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )
        else:
            from diffusers import DiffusionPipeline
            self._pipe = DiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype,
            )

    def _get_pipeline_type(self, model_path: str = None, is_single_file: bool = False) -> str:
        """
        Get the pipeline type for the current model.

        Args:
            model_path: Path to model (used for custom models)
            is_single_file: Whether it's a single file checkpoint

        Returns:
            Pipeline type string: "flux", "zimage", "sdxl", "sd15", or "auto"
        """
        # Known models
        if self._settings.name in ["flux-schnell", "flux-dev"]:
            return "flux"
        elif self._settings.name == "z-turbo":
            return "zimage"
        elif self._settings.name == "sdxl":
            return "sdxl"
        elif self._settings.name == "custom":
            # Check if user explicitly set pipeline type
            if self._settings.pipeline_type and self._settings.pipeline_type != "auto":
                return self._settings.pipeline_type

            # Try to detect from metadata or model structure
            return self._detect_pipeline_type(model_path, is_single_file)
        else:
            raise ValueError(f"Unknown model: {self._settings.name}")

    def _detect_pipeline_type(self, model_path: str, is_single_file: bool) -> str:
        """
        Auto-detect pipeline type from model path.

        Args:
            model_path: Path to model
            is_single_file: Whether it's a single file checkpoint

        Returns:
            Detected pipeline type
        """
        path = Path(model_path)

        # Check for CivitAI metadata
        if path.parent.name.startswith(("civitai", "url")):
            metadata_path = path.parent / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path) as f:
                    metadata = json.load(f)
                base_model = metadata.get("base_model", "")
                if base_model:
                    base_lower = base_model.lower()
                    if "xl" in base_lower or "sdxl" in base_lower:
                        logger.info(f"Detected SDXL model from metadata")
                        return "sdxl"
                    elif "1.5" in base_lower or "sd 1" in base_lower:
                        logger.info(f"Detected SD 1.5 model from metadata")
                        return "sd15"
                    elif "flux" in base_lower:
                        logger.info(f"Detected FLUX model from metadata")
                        return "flux"

        # Check for diffusers format model_index.json
        if path.is_dir():
            model_index_path = path / "model_index.json"
            if model_index_path.exists():
                with open(model_index_path) as f:
                    index = json.load(f)
                class_name = index.get("_class_name", "")
                if "XL" in class_name:
                    return "sdxl"
                elif "Flux" in class_name:
                    return "flux"
                elif "StableDiffusion" in class_name:
                    return "sd15"

        # Check filename patterns
        filename = path.name.lower()
        if "xl" in filename or "sdxl" in filename:
            logger.info(f"Detected SDXL model from filename")
            return "sdxl"
        elif "flux" in filename:
            logger.info(f"Detected FLUX model from filename")
            return "flux"
        elif "sd15" in filename or "sd_1" in filename or "1.5" in filename:
            logger.info(f"Detected SD 1.5 model from filename")
            return "sd15"

        # Default to SDXL as most CivitAI models are SDXL
        logger.info(f"Could not detect pipeline type, defaulting to SDXL")
        return "sdxl"

    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            if self._device == "cuda":
                import torch  # Lazy import
                torch.cuda.empty_cache()
            logger.info("Model unloaded")

    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 1024,
        height: int = 1024,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        output_format: str = "png",
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> GenerationResult:
        """
        Generate an image from a prompt.

        Args:
            prompt: Text description of the image
            negative_prompt: What not to include
            width: Image width
            height: Image height
            steps: Number of inference steps
            guidance_scale: Classifier-free guidance scale
            seed: Random seed for reproducibility
            output_format: Output format (png, jpeg, webp)
            progress_callback: Callback for progress updates (step, total)

        Returns:
            GenerationResult with the generated image
        """
        import torch  # Lazy import

        if not self.is_loaded:
            raise RuntimeError("Model not loaded")

        warnings = []
        start = time.time()

        # Round dimensions to multiple of 64
        original_width, original_height = width, height
        width = self._round_to_multiple(width, 64)
        height = self._round_to_multiple(height, 64)

        if width != original_width:
            warnings.append(f"width adjusted from {original_width} to {width} (multiple of 64)")
        if height != original_height:
            warnings.append(f"height adjusted from {original_height} to {height} (multiple of 64)")

        # Use defaults if not specified
        if steps is None:
            steps = self._settings.default_steps
        else:
            # Clamp to optimal range with warning
            min_steps, max_steps = self._settings.optimal_steps_range
            if steps < min_steps:
                warnings.append(f"steps adjusted from {steps} to {min_steps} (model minimum)")
                steps = min_steps
            elif steps > max_steps:
                warnings.append(f"steps adjusted from {steps} to {max_steps} (model maximum)")
                steps = max_steps

        if guidance_scale is None:
            guidance_scale = self._settings.default_guidance_scale

        # Generate seed if not provided
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator(device=self._device).manual_seed(seed)

        # Build generation kwargs
        gen_kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "generator": generator,
        }

        # Add guidance scale if model uses it
        if guidance_scale > 0:
            gen_kwargs["guidance_scale"] = guidance_scale

        # Add negative prompt for SDXL
        if negative_prompt and self._settings.name == "sdxl":
            gen_kwargs["negative_prompt"] = negative_prompt

        # Add callback for progress
        if progress_callback:
            def callback_wrapper(pipe, step, timestep, callback_kwargs):
                progress_callback(step + 1, steps)
                return callback_kwargs

            gen_kwargs["callback_on_step_end"] = callback_wrapper

        # Generate image
        with torch.inference_mode():
            result = self._pipe(**gen_kwargs)
            image = result.images[0]

        # Convert to bytes
        buffer = io.BytesIO()
        format_map = {"png": "PNG", "jpeg": "JPEG", "webp": "WEBP"}
        img_format = format_map.get(output_format, "PNG")

        save_kwargs = {}
        if img_format == "JPEG":
            save_kwargs["quality"] = 95

        image.save(buffer, format=img_format, **save_kwargs)
        buffer.seek(0)
        image_bytes = buffer.getvalue()

        generation_time = time.time() - start
        logger.info(f"Image generated in {generation_time:.2f}s (seed={seed})")

        return GenerationResult(
            image=image,
            image_bytes=image_bytes,
            seed=seed,
            generation_time=generation_time,
            output_format=output_format,
            warnings=warnings,
        )

    @staticmethod
    def _round_to_multiple(value: int, multiple: int) -> int:
        """Round a value to the nearest multiple."""
        return ((value + multiple // 2) // multiple) * multiple

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        """Hash a prompt for logging/analytics (privacy-preserving)."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get_gpu_info(self) -> dict:
        """Get GPU information."""
        try:
            import torch  # Lazy import
        except ImportError:
            return {
                "available": False,
                "device": self._device,
                "error": "torch not installed",
            }

        if self._device != "cuda" or not torch.cuda.is_available():
            return {
                "available": False,
                "device": self._device,
            }

        try:
            props = torch.cuda.get_device_properties(0)
            memory_allocated = torch.cuda.memory_allocated(0)
            memory_reserved = torch.cuda.memory_reserved(0)
            total_memory = props.total_memory

            return {
                "available": True,
                "device": self._device,
                "name": props.name,
                "total_memory_gb": round(total_memory / (1024**3), 2),
                "allocated_memory_gb": round(memory_allocated / (1024**3), 2),
                "reserved_memory_gb": round(memory_reserved / (1024**3), 2),
                "free_memory_gb": round((total_memory - memory_reserved) / (1024**3), 2),
            }
        except Exception as e:
            logger.error(f"Failed to get GPU info: {e}")
            return {
                "available": True,
                "device": self._device,
                "error": str(e),
            }
