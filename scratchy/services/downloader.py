"""Model download service for CivitAI and direct URLs."""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from urllib.parse import urlparse, unquote

import httpx

logger = logging.getLogger(__name__)

# CivitAI API base URL
CIVITAI_API_BASE = "https://civitai.com/api/v1"

# Common file extensions for model files
MODEL_EXTENSIONS = {".safetensors", ".ckpt", ".pt", ".bin", ".pth"}

# Default chunk size for downloads (1MB)
DEFAULT_CHUNK_SIZE = 1024 * 1024


@dataclass
class CivitAIModelInfo:
    """Information about a CivitAI model."""
    model_id: str
    model_name: str
    version_id: str
    version_name: str
    download_url: str
    filename: str
    size_bytes: int
    sha256: Optional[str] = None
    trigger_words: list[str] = field(default_factory=list)
    base_model: Optional[str] = None  # e.g., "SDXL 1.0", "SD 1.5"
    model_type: Optional[str] = None  # e.g., "Checkpoint", "LORA"


@dataclass
class CachedModel:
    """Information about a cached model."""
    name: str
    path: Path
    size_bytes: int
    source: str  # "civitai", "url", "local"
    metadata: dict = field(default_factory=dict)


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    downloaded_bytes: int
    total_bytes: int
    speed_bytes_per_sec: float
    eta_seconds: Optional[float]


class ModelDownloader:
    """Handles downloads from CivitAI and direct URLs."""

    def __init__(self, models_dir: Path, civitai_api_key: Optional[str] = None):
        """
        Initialize the downloader.

        Args:
            models_dir: Directory to store downloaded models
            civitai_api_key: Optional CivitAI API key for faster downloads
        """
        self.models_dir = Path(models_dir)
        self.civitai_api_key = civitai_api_key or os.environ.get("CIVITAI_API_KEY")
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        (self.models_dir / "civitai").mkdir(exist_ok=True)
        (self.models_dir / "url").mkdir(exist_ok=True)

    def _get_http_client(self) -> httpx.Client:
        """Get an HTTP client with appropriate headers."""
        headers = {
            "User-Agent": "Scratchy/2.0 (https://github.com/scratchy/scratchy)"
        }
        if self.civitai_api_key:
            headers["Authorization"] = f"Bearer {self.civitai_api_key}"
        return httpx.Client(headers=headers, follow_redirects=True, timeout=30.0)

    def parse_civitai_url(self, url: str) -> tuple[str, Optional[str]]:
        """
        Parse a CivitAI URL to extract model ID and optional version ID.

        Args:
            url: CivitAI URL (e.g., "https://civitai.com/models/12345" or
                 "https://civitai.com/models/12345?modelVersionId=67890")

        Returns:
            Tuple of (model_id, version_id or None)
        """
        # Handle direct model ID input
        if url.isdigit():
            return url, None

        parsed = urlparse(url)

        # Extract model ID from path: /models/12345 or /models/12345/anything
        path_match = re.match(r"/models/(\d+)", parsed.path)
        if not path_match:
            raise ValueError(f"Invalid CivitAI URL: {url}")

        model_id = path_match.group(1)

        # Check for version ID in query params
        version_id = None
        if parsed.query:
            params = dict(p.split("=") for p in parsed.query.split("&") if "=" in p)
            version_id = params.get("modelVersionId")

        return model_id, version_id

    def get_civitai_model_info(
        self, model_id: str, version_id: Optional[str] = None
    ) -> CivitAIModelInfo:
        """
        Get model information from CivitAI API.

        Args:
            model_id: CivitAI model ID
            version_id: Optional specific version ID (uses latest if not specified)

        Returns:
            CivitAIModelInfo with download URL and metadata
        """
        with self._get_http_client() as client:
            # Get model info
            response = client.get(f"{CIVITAI_API_BASE}/models/{model_id}")
            response.raise_for_status()
            model_data = response.json()

            model_name = model_data.get("name", f"model_{model_id}")
            model_type = model_data.get("type")
            versions = model_data.get("modelVersions", [])

            if not versions:
                raise ValueError(f"No versions found for model {model_id}")

            # Find the requested version or use latest
            if version_id:
                version = next(
                    (v for v in versions if str(v.get("id")) == str(version_id)),
                    None
                )
                if not version:
                    raise ValueError(f"Version {version_id} not found for model {model_id}")
            else:
                version = versions[0]  # Latest version

            version_name = version.get("name", "")
            version_id = str(version.get("id"))
            base_model = version.get("baseModel")
            trigger_words = version.get("trainedWords", [])

            # Find the primary file (usually safetensors)
            files = version.get("files", [])
            if not files:
                raise ValueError(f"No files found for version {version_id}")

            # Prefer safetensors, then ckpt
            primary_file = None
            for f in files:
                if f.get("name", "").endswith(".safetensors"):
                    primary_file = f
                    break
            if not primary_file:
                for f in files:
                    if f.get("name", "").endswith(".ckpt"):
                        primary_file = f
                        break
            if not primary_file:
                primary_file = files[0]

            filename = primary_file.get("name", f"model_{model_id}.safetensors")
            download_url = primary_file.get("downloadUrl", "")
            size_bytes = primary_file.get("sizeKB", 0) * 1024

            # Get hash if available
            hashes = primary_file.get("hashes", {})
            sha256 = hashes.get("SHA256")

            return CivitAIModelInfo(
                model_id=model_id,
                model_name=model_name,
                version_id=version_id,
                version_name=version_name,
                download_url=download_url,
                filename=filename,
                size_bytes=int(size_bytes),
                sha256=sha256,
                trigger_words=trigger_words,
                base_model=base_model,
                model_type=model_type,
            )

    def download_from_civitai(
        self,
        model_id: str,
        version_id: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> Path:
        """
        Download a model from CivitAI.

        Args:
            model_id: CivitAI model ID
            version_id: Optional specific version ID
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the downloaded model file
        """
        info = self.get_civitai_model_info(model_id, version_id)
        logger.info(f"Downloading from CivitAI: {info.model_name} v{info.version_name}")

        # Create subdirectory for this model
        model_dir = self.models_dir / "civitai" / f"{model_id}_{info.version_id}"
        model_dir.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / info.filename

        # Check if already downloaded and valid
        if model_path.exists():
            if info.sha256 and self._verify_hash(model_path, info.sha256):
                logger.info(f"Model already downloaded and verified: {model_path}")
                self._save_metadata(model_dir, info)
                return model_path
            elif not info.sha256:
                logger.info(f"Model already downloaded (no hash to verify): {model_path}")
                self._save_metadata(model_dir, info)
                return model_path

        # Download with resume support
        self._download_file(
            url=info.download_url,
            dest_path=model_path,
            expected_size=info.size_bytes,
            expected_hash=info.sha256,
            progress_callback=progress_callback,
        )

        # Save metadata
        self._save_metadata(model_dir, info)

        logger.info(f"Downloaded model to: {model_path}")
        return model_path

    def download_from_url(
        self,
        url: str,
        filename: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> Path:
        """
        Download a model from a direct URL.

        Args:
            url: URL to download from
            filename: Optional filename (extracted from URL/headers if not provided)
            progress_callback: Optional callback for progress updates

        Returns:
            Path to the downloaded model file
        """
        logger.info(f"Downloading from URL: {url}")

        # Create a safe directory name from URL
        url_hash = hashlib.sha256(url.encode()).hexdigest()[:12]
        model_dir = self.models_dir / "url" / url_hash
        model_dir.mkdir(parents=True, exist_ok=True)

        # Get filename from Content-Disposition or URL if not provided
        if not filename:
            filename = self._get_filename_from_url(url)

        model_path = model_dir / filename

        # Check if already downloaded
        if model_path.exists():
            logger.info(f"Model already downloaded: {model_path}")
            return model_path

        # Download
        self._download_file(
            url=url,
            dest_path=model_path,
            progress_callback=progress_callback,
        )

        # Save basic metadata
        metadata = {
            "source": "url",
            "url": url,
            "filename": filename,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Downloaded model to: {model_path}")
        return model_path

    def _get_filename_from_url(self, url: str) -> str:
        """Extract filename from URL or Content-Disposition header."""
        # First, try to get it from URL path
        parsed = urlparse(url)
        path_filename = Path(unquote(parsed.path)).name

        # If it looks like a model file, use it
        if any(path_filename.endswith(ext) for ext in MODEL_EXTENSIONS):
            return path_filename

        # Otherwise, try HEAD request to get Content-Disposition
        try:
            with self._get_http_client() as client:
                response = client.head(url)
                cd = response.headers.get("content-disposition", "")
                if "filename=" in cd:
                    match = re.search(r'filename="?([^";\n]+)"?', cd)
                    if match:
                        return match.group(1)
        except Exception:
            pass

        # Fallback to URL path or generic name
        if path_filename and "." in path_filename:
            return path_filename
        return "model.safetensors"

    def _download_file(
        self,
        url: str,
        dest_path: Path,
        expected_size: Optional[int] = None,
        expected_hash: Optional[str] = None,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> None:
        """
        Download a file with resume support.

        Args:
            url: URL to download from
            dest_path: Destination file path
            expected_size: Expected file size (for validation)
            expected_hash: Expected SHA256 hash (for validation)
            progress_callback: Optional callback for progress updates
        """
        import time

        partial_path = dest_path.with_suffix(dest_path.suffix + ".partial")

        # Check for partial download
        resume_pos = 0
        if partial_path.exists():
            resume_pos = partial_path.stat().st_size
            logger.info(f"Resuming download from byte {resume_pos}")

        headers = {}
        if resume_pos > 0:
            headers["Range"] = f"bytes={resume_pos}-"

        with self._get_http_client() as client:
            # Use stream for large file downloads
            with client.stream("GET", url, headers=headers, timeout=None) as response:
                if response.status_code == 416:  # Range not satisfiable - file complete
                    if partial_path.exists():
                        partial_path.rename(dest_path)
                    return

                response.raise_for_status()

                # Get total size
                total_size = expected_size
                if "content-length" in response.headers:
                    content_length = int(response.headers["content-length"])
                    if response.status_code == 206:  # Partial content
                        total_size = resume_pos + content_length
                    else:
                        total_size = content_length
                        resume_pos = 0  # Server doesn't support range

                # Download with progress tracking
                start_time = time.time()
                downloaded = resume_pos

                mode = "ab" if resume_pos > 0 and response.status_code == 206 else "wb"
                with open(partial_path, mode) as f:
                    for chunk in response.iter_bytes(chunk_size=DEFAULT_CHUNK_SIZE):
                        f.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size:
                            elapsed = time.time() - start_time
                            speed = (downloaded - resume_pos) / elapsed if elapsed > 0 else 0
                            eta = (total_size - downloaded) / speed if speed > 0 else None
                            progress_callback(DownloadProgress(
                                downloaded_bytes=downloaded,
                                total_bytes=total_size,
                                speed_bytes_per_sec=speed,
                                eta_seconds=eta,
                            ))

        # Verify hash if provided
        if expected_hash:
            if not self._verify_hash(partial_path, expected_hash):
                partial_path.unlink()
                raise ValueError("Downloaded file hash mismatch")

        # Move to final location
        partial_path.rename(dest_path)

    def _verify_hash(self, file_path: Path, expected_hash: str) -> bool:
        """Verify file SHA256 hash."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(DEFAULT_CHUNK_SIZE), b""):
                sha256.update(chunk)
        actual_hash = sha256.hexdigest().upper()
        return actual_hash == expected_hash.upper()

    def _save_metadata(self, model_dir: Path, info: CivitAIModelInfo) -> None:
        """Save model metadata to JSON file."""
        metadata = {
            "source": "civitai",
            "model_id": info.model_id,
            "model_name": info.model_name,
            "version_id": info.version_id,
            "version_name": info.version_name,
            "filename": info.filename,
            "size_bytes": info.size_bytes,
            "sha256": info.sha256,
            "trigger_words": info.trigger_words,
            "base_model": info.base_model,
            "model_type": info.model_type,
        }
        with open(model_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def get_cached_models(self) -> list[CachedModel]:
        """
        Get list of all cached models.

        Returns:
            List of CachedModel objects
        """
        models = []

        # Scan civitai directory
        civitai_dir = self.models_dir / "civitai"
        if civitai_dir.exists():
            for model_dir in civitai_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                        # Find the model file
                        model_files = [
                            f for f in model_dir.iterdir()
                            if f.suffix in MODEL_EXTENSIONS
                        ]
                        if model_files:
                            model_file = model_files[0]
                            models.append(CachedModel(
                                name=metadata.get("model_name", model_dir.name),
                                path=model_file,
                                size_bytes=model_file.stat().st_size,
                                source="civitai",
                                metadata=metadata,
                            ))

        # Scan url directory
        url_dir = self.models_dir / "url"
        if url_dir.exists():
            for model_dir in url_dir.iterdir():
                if model_dir.is_dir():
                    metadata_path = model_dir / "metadata.json"
                    metadata = {}
                    if metadata_path.exists():
                        with open(metadata_path) as f:
                            metadata = json.load(f)
                    # Find the model file
                    model_files = [
                        f for f in model_dir.iterdir()
                        if f.suffix in MODEL_EXTENSIONS
                    ]
                    if model_files:
                        model_file = model_files[0]
                        models.append(CachedModel(
                            name=metadata.get("filename", model_file.name),
                            path=model_file,
                            size_bytes=model_file.stat().st_size,
                            source="url",
                            metadata=metadata,
                        ))

        return models

    def get_model_path(self, identifier: str) -> Optional[Path]:
        """
        Get path to a cached model by identifier.

        Args:
            identifier: Model name, CivitAI ID, or partial path

        Returns:
            Path to model file or None if not found
        """
        for model in self.get_cached_models():
            # Match by name
            if model.name.lower() == identifier.lower():
                return model.path
            # Match by CivitAI model ID
            if model.source == "civitai":
                if model.metadata.get("model_id") == identifier:
                    return model.path
            # Match by path contains
            if identifier.lower() in str(model.path).lower():
                return model.path

        return None

    def remove_model(self, identifier: str) -> bool:
        """
        Remove a cached model.

        Args:
            identifier: Model name, ID, or path

        Returns:
            True if removed, False if not found
        """
        import shutil

        for model in self.get_cached_models():
            match = False
            if model.name.lower() == identifier.lower():
                match = True
            elif model.source == "civitai" and model.metadata.get("model_id") == identifier:
                match = True
            elif identifier.lower() in str(model.path).lower():
                match = True

            if match:
                # Remove the entire model directory
                model_dir = model.path.parent
                shutil.rmtree(model_dir)
                logger.info(f"Removed model: {model.name}")
                return True

        return False

    def get_model_info(self, identifier: str) -> Optional[CachedModel]:
        """
        Get information about a cached model.

        Args:
            identifier: Model name, ID, or path

        Returns:
            CachedModel or None if not found
        """
        for model in self.get_cached_models():
            if model.name.lower() == identifier.lower():
                return model
            if model.source == "civitai" and model.metadata.get("model_id") == identifier:
                return model
            if identifier.lower() in str(model.path).lower():
                return model
        return None
