"""
Configuration management with hierarchical loading:
1. Code defaults
2. config.yaml file
3. Environment variables (highest priority)
"""

import os
from pathlib import Path
from typing import Optional, Literal
from functools import lru_cache

import yaml
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ServerSettings(BaseSettings):
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1


class ModelSettings(BaseSettings):
    """Model configuration."""
    name: Literal["flux-schnell", "flux-dev", "z-turbo", "sdxl"] = "flux-schnell"
    quantization: Literal["none", "8bit", "4bit"] = "none"
    device: Literal["cuda", "mps", "cpu"] = "cuda"
    skip_load: bool = False  # Set to True to skip model loading (for testing API without ML)

    @property
    def model_id(self) -> str:
        """Get the HuggingFace model ID."""
        model_map = {
            "flux-schnell": "black-forest-labs/FLUX.1-schnell",
            "flux-dev": "black-forest-labs/FLUX.1-dev",
            "z-turbo": "Tongyi-MAI/Z-Image-Turbo",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        }
        return model_map[self.name]

    @property
    def default_steps(self) -> int:
        """Get default inference steps for the model."""
        steps_map = {
            "flux-schnell": 4,
            "flux-dev": 28,
            "z-turbo": 8,
            "sdxl": 30,
        }
        return steps_map[self.name]

    @property
    def default_guidance_scale(self) -> float:
        """Get default guidance scale for the model."""
        guidance_map = {
            "flux-schnell": 0.0,
            "flux-dev": 3.5,
            "z-turbo": 1.0,
            "sdxl": 7.5,
        }
        return guidance_map[self.name]

    @property
    def optimal_steps_range(self) -> tuple[int, int]:
        """Get optimal step range for the model."""
        ranges = {
            "flux-schnell": (1, 8),
            "flux-dev": (20, 50),
            "z-turbo": (4, 12),
            "sdxl": (20, 50),
        }
        return ranges[self.name]


class QueueSettings(BaseSettings):
    """Queue configuration."""
    max_depth: int = Field(default=10, ge=1, le=100)


class StorageSettings(BaseSettings):
    """Storage configuration."""
    jobs_dir: Path = Path("/var/scratchy/jobs")
    jobs_ttl_hours: int = Field(default=1, ge=1, le=24)
    db_path: Path = Path("/var/scratchy/scratchy.db")
    backup_dir: Path = Path("/var/scratchy/backups")
    backup_retention_days: int = Field(default=7, ge=1, le=30)

    @field_validator("jobs_dir", "db_path", "backup_dir", mode="before")
    @classmethod
    def expand_path(cls, v):
        if isinstance(v, str):
            return Path(v).expanduser()
        return v


class AuthSettings(BaseSettings):
    """Authentication configuration."""
    default_rate_limit: int = Field(default=10, ge=1, le=1000)  # requests per minute
    admin_key: Optional[str] = None  # Optional admin API key for admin endpoints


class LoggingSettings(BaseSettings):
    """Logging configuration."""
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    format: Literal["json", "text"] = "json"
    include_prompts: bool = False
    retention_days: int = Field(default=7, ge=1, le=90)


class SecuritySettings(BaseSettings):
    """Security configuration."""
    cors_origins: list[str] = ["*"]
    max_prompt_length: int = Field(default=2000, ge=100, le=10000)
    max_negative_prompt_length: int = Field(default=500, ge=50, le=2000)


class Settings(BaseSettings):
    """Main application settings."""

    model_config = SettingsConfigDict(
        env_prefix="SCRATCHY_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerSettings = Field(default_factory=ServerSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    auth: AuthSettings = Field(default_factory=AuthSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)


def load_yaml_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return {}

    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: dict, override: dict) -> dict:
    """Deep merge two configuration dictionaries."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings with hierarchical loading:
    1. Code defaults (in Pydantic models)
    2. config.yaml file
    3. Environment variables
    """
    # Look for config.yaml in several locations
    config_paths = [
        Path("config.yaml"),
        Path("config.yml"),
        Path("/etc/scratchy/config.yaml"),
        Path.home() / ".config" / "scratchy" / "config.yaml",
    ]

    yaml_config = {}
    for path in config_paths:
        if path.exists():
            yaml_config = load_yaml_config(path)
            break

    # Environment variables are handled automatically by pydantic-settings
    # They override both defaults and YAML config

    # Create settings with YAML config as initial values
    # Environment variables will override via pydantic-settings
    if yaml_config:
        # Flatten nested config for environment variable style
        settings = Settings(**yaml_config)
    else:
        settings = Settings()

    return settings
