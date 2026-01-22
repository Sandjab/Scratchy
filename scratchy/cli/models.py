#!/usr/bin/env python3
"""CLI for model management - download, list, remove models."""

import argparse
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scratchy.config import get_settings

# Import downloader directly to avoid triggering __init__.py imports
# which may fail if database models aren't set up
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "downloader",
    Path(__file__).parent.parent / "services" / "downloader.py"
)
_downloader_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_downloader_module)
ModelDownloader = _downloader_module.ModelDownloader
DownloadProgress = _downloader_module.DownloadProgress


def format_size(size_bytes: int) -> str:
    """Format size in human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_eta(seconds: float) -> str:
    """Format ETA in human-readable format."""
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.0f}m"
    else:
        return f"{seconds / 3600:.1f}h"


def create_progress_bar(progress: DownloadProgress) -> str:
    """Create a text progress bar."""
    width = 30
    pct = progress.downloaded_bytes / progress.total_bytes if progress.total_bytes else 0
    filled = int(width * pct)
    bar = "=" * filled + "-" * (width - filled)
    speed = format_size(progress.speed_bytes_per_sec) + "/s"
    eta = format_eta(progress.eta_seconds)
    downloaded = format_size(progress.downloaded_bytes)
    total = format_size(progress.total_bytes) if progress.total_bytes else "?"
    return f"[{bar}] {pct*100:.1f}% | {downloaded}/{total} | {speed} | ETA: {eta}"


def download_model(args):
    """Download a model from CivitAI or direct URL."""
    settings = get_settings()
    downloader = ModelDownloader(settings.storage.models_dir)

    source = args.source

    # Determine if it's a CivitAI URL/ID or direct URL
    is_civitai = False
    if source.isdigit():
        is_civitai = True
    elif "civitai.com" in source:
        is_civitai = True
    elif source.startswith("http"):
        is_civitai = False
    else:
        print(f"Error: Cannot determine source type for: {source}")
        print("Provide a CivitAI URL, model ID, or direct download URL")
        sys.exit(1)

    # Progress callback with tqdm fallback to simple print
    try:
        from tqdm import tqdm
        pbar = None

        def progress_callback(progress: DownloadProgress):
            nonlocal pbar
            if pbar is None and progress.total_bytes:
                pbar = tqdm(
                    total=progress.total_bytes,
                    unit="B",
                    unit_scale=True,
                    desc="Downloading",
                )
            if pbar:
                pbar.update(progress.downloaded_bytes - pbar.n)

    except ImportError:
        last_pct = [0]

        def progress_callback(progress: DownloadProgress):
            pct = int(progress.downloaded_bytes / progress.total_bytes * 100) if progress.total_bytes else 0
            if pct >= last_pct[0] + 5:  # Update every 5%
                print(create_progress_bar(progress), end="\r")
                last_pct[0] = pct

    try:
        if is_civitai:
            # Parse CivitAI URL or use ID directly
            if source.isdigit():
                model_id = source
                version_id = args.version
            else:
                model_id, version_id = downloader.parse_civitai_url(source)
                if args.version:
                    version_id = args.version

            # Get model info first
            print(f"\nFetching model info from CivitAI...")
            info = downloader.get_civitai_model_info(model_id, version_id)
            print(f"\nModel: {info.model_name}")
            print(f"Version: {info.version_name}")
            print(f"Type: {info.model_type or 'Unknown'}")
            print(f"Base Model: {info.base_model or 'Unknown'}")
            print(f"File: {info.filename}")
            print(f"Size: {format_size(info.size_bytes)}")
            if info.trigger_words:
                print(f"Trigger Words: {', '.join(info.trigger_words)}")
            print()

            # Download
            model_path = downloader.download_from_civitai(
                model_id=model_id,
                version_id=version_id,
                progress_callback=progress_callback,
            )

        else:
            # Direct URL download
            print(f"\nDownloading from URL...")
            print(f"URL: {source}")
            if args.name:
                print(f"Filename: {args.name}")
            print()

            model_path = downloader.download_from_url(
                url=source,
                filename=args.name,
                progress_callback=progress_callback,
            )

        print(f"\n\nDownload complete!")
        print(f"Model saved to: {model_path}")

        # Show config example
        print("\nTo use this model, add to your config.yaml:")
        print("-" * 40)
        print("model:")
        print('  name: "custom"')
        print(f'  local_path: "{model_path}"')
        print("-" * 40)

    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


def list_models(args):
    """List all downloaded models."""
    settings = get_settings()
    downloader = ModelDownloader(settings.storage.models_dir)

    models = downloader.get_cached_models()

    if not models:
        print("\nNo models downloaded yet.")
        print(f"\nModels directory: {settings.storage.models_dir}")
        print("\nTo download a model:")
        print("  scratchy-models download https://civitai.com/models/12345")
        print("  scratchy-models download 12345")
        print("  scratchy-models download https://example.com/model.safetensors")
        return

    print(f"\n{'Name':<40} {'Source':<10} {'Size':<12} {'Path'}")
    print("-" * 100)

    total_size = 0
    for model in models:
        name = model.name[:38] + ".." if len(model.name) > 40 else model.name
        size = format_size(model.size_bytes)
        total_size += model.size_bytes
        path = str(model.path)
        if len(path) > 40:
            path = "..." + path[-37:]
        print(f"{name:<40} {model.source:<10} {size:<12} {path}")

    print("-" * 100)
    print(f"Total: {len(models)} models ({format_size(total_size)})")
    print(f"Models directory: {settings.storage.models_dir}\n")


def show_model_info(args):
    """Show detailed information about a model."""
    settings = get_settings()
    downloader = ModelDownloader(settings.storage.models_dir)

    model = downloader.get_model_info(args.model)

    if not model:
        print(f"Error: Model not found: {args.model}")
        print("\nUse 'scratchy-models list' to see available models")
        sys.exit(1)

    print(f"\nModel Information")
    print("=" * 50)
    print(f"Name:        {model.name}")
    print(f"Source:      {model.source}")
    print(f"Path:        {model.path}")
    print(f"Size:        {format_size(model.size_bytes)}")

    if model.metadata:
        print(f"\nMetadata:")
        print("-" * 50)

        if model.source == "civitai":
            print(f"  Model ID:      {model.metadata.get('model_id', 'N/A')}")
            print(f"  Version ID:    {model.metadata.get('version_id', 'N/A')}")
            print(f"  Version Name:  {model.metadata.get('version_name', 'N/A')}")
            print(f"  Model Type:    {model.metadata.get('model_type', 'N/A')}")
            print(f"  Base Model:    {model.metadata.get('base_model', 'N/A')}")
            trigger_words = model.metadata.get("trigger_words", [])
            if trigger_words:
                print(f"  Trigger Words: {', '.join(trigger_words)}")
            sha256 = model.metadata.get("sha256")
            if sha256:
                print(f"  SHA256:        {sha256[:16]}...")
        elif model.source == "url":
            print(f"  URL:           {model.metadata.get('url', 'N/A')}")
            print(f"  Filename:      {model.metadata.get('filename', 'N/A')}")

    print(f"\nConfig Example:")
    print("-" * 50)
    print("model:")
    print('  name: "custom"')
    print(f'  local_path: "{model.path}"')

    # Add pipeline hint based on metadata
    if model.metadata:
        base_model = model.metadata.get("base_model", "")
        if base_model:
            if "xl" in base_model.lower():
                print('  pipeline_type: "sdxl"  # Auto-detected')
            elif "1.5" in base_model or "sd 1" in base_model.lower():
                print('  pipeline_type: "sd15"  # Auto-detected')
            elif "flux" in base_model.lower():
                print('  pipeline_type: "flux"  # Auto-detected')

    print()


def remove_model(args):
    """Remove a downloaded model."""
    settings = get_settings()
    downloader = ModelDownloader(settings.storage.models_dir)

    # Get model info first
    model = downloader.get_model_info(args.model)
    if not model:
        print(f"Error: Model not found: {args.model}")
        print("\nUse 'scratchy-models list' to see available models")
        sys.exit(1)

    if not args.yes:
        print(f"\nAbout to remove:")
        print(f"  Name: {model.name}")
        print(f"  Path: {model.path}")
        print(f"  Size: {format_size(model.size_bytes)}")
        confirm = input("\nAre you sure? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return

    success = downloader.remove_model(args.model)
    if success:
        print(f"Removed: {model.name}")
    else:
        print(f"Error: Failed to remove model")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Scratchy Model Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download from CivitAI
  scratchy-models download https://civitai.com/models/12345
  scratchy-models download 12345 --version 67890

  # Download from direct URL
  scratchy-models download https://example.com/model.safetensors --name my_model

  # List downloaded models
  scratchy-models list

  # Show model info
  scratchy-models info my_model

  # Remove a model
  scratchy-models remove my_model
""",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Download command
    download_parser = subparsers.add_parser(
        "download",
        help="Download a model from CivitAI or URL"
    )
    download_parser.add_argument(
        "source",
        help="CivitAI URL, model ID, or direct download URL"
    )
    download_parser.add_argument(
        "--version", "-v",
        help="CivitAI version ID (optional, uses latest if not specified)"
    )
    download_parser.add_argument(
        "--name", "-n",
        help="Filename for URL downloads (auto-detected if not specified)"
    )
    download_parser.set_defaults(func=download_model)

    # List command
    list_parser = subparsers.add_parser("list", help="List downloaded models")
    list_parser.set_defaults(func=list_models)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model", help="Model name or ID")
    info_parser.set_defaults(func=show_model_info)

    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a downloaded model")
    remove_parser.add_argument("model", help="Model name or ID")
    remove_parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt"
    )
    remove_parser.set_defaults(func=remove_model)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
