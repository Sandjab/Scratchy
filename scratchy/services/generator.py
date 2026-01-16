"""Image generation service."""

import io
import hashlib
import logging
import time
from typing import Optional, Callable, Any, TYPE_CHECKING
from dataclasses import dataclass

from PIL import Image

from scratchy.config import ModelSettings

# Lazy import torch to avoid loading ML dependencies if not needed
if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)


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

        # Load appropriate pipeline
        pipeline_type = self._get_pipeline_type()

        if pipeline_type == "flux":
            from diffusers import FluxPipeline
            self._pipe = FluxPipeline.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
            )
        elif pipeline_type == "zimage":
            from diffusers import DiffusionPipeline
            self._pipe = DiffusionPipeline.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
            )
        elif pipeline_type == "sdxl":
            from diffusers import StableDiffusionXLPipeline
            self._pipe = StableDiffusionXLPipeline.from_pretrained(
                self._model_name,
                torch_dtype=dtype,
                use_safetensors=True,
                variant="fp16" if dtype == torch.float16 else None,
            )

        # Move to device
        self._pipe.to(self._device)

        # Apply memory optimizations
        if hasattr(self._pipe, 'enable_attention_slicing'):
            self._pipe.enable_attention_slicing()

        if hasattr(self._pipe, 'enable_vae_slicing'):
            self._pipe.enable_vae_slicing()

        elapsed = time.time() - start
        logger.info(f"Model loaded in {elapsed:.1f}s")

    def _get_pipeline_type(self) -> str:
        """Get the pipeline type for the current model."""
        if self._settings.name in ["flux-schnell", "flux-dev"]:
            return "flux"
        elif self._settings.name == "z-turbo":
            return "zimage"
        elif self._settings.name == "sdxl":
            return "sdxl"
        else:
            raise ValueError(f"Unknown model: {self._settings.name}")

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
