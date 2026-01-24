#!/usr/bin/env python3
"""
Environment check script for Scratchy.

Verifies Python version, PyTorch installation, and GPU availability.
Supports both NVIDIA CUDA and Apple Metal (MPS) backends.
"""

import sys
import platform


def get_system_info():
    """Get basic system information."""
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Machine: {platform.machine()}")
    print(f"Python version: {sys.version}")
    print(f"Python version info: {sys.version_info}")

    # macOS specific info
    if platform.system() == "Darwin":
        print(f"macOS version: {platform.mac_ver()[0]}")
        # Try to get chip info
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"CPU: {result.stdout.strip()}")

            # Get memory info
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                mem_bytes = int(result.stdout.strip())
                mem_gb = mem_bytes / (1024 ** 3)
                print(f"Total RAM: {mem_gb:.1f} GB")
        except Exception as e:
            print(f"Could not get detailed system info: {e}")


def check_pytorch():
    """Check PyTorch installation and versions."""
    print("\n" + "=" * 60)
    print("PYTORCH INSTALLATION")
    print("=" * 60)

    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"PyTorch built with CUDA: {torch.version.cuda or 'No'}")
    except ImportError:
        print("ERROR: PyTorch is not installed!")
        print("Install with: pip install torch")
        return None

    try:
        import torchvision
        print(f"torchvision version: {torchvision.__version__}")
    except ImportError:
        print("torchvision: not installed")

    try:
        import torchaudio
        print(f"torchaudio version: {torchaudio.__version__}")
    except ImportError:
        print("torchaudio: not installed")

    return torch


def check_cuda(torch):
    """Check CUDA availability and GPU info."""
    print("\n" + "=" * 60)
    print("CUDA / NVIDIA GPU")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("CUDA available: No")
        if platform.system() == "Darwin":
            print("  (This is expected on macOS - use MPS instead)")
        else:
            print("  Check NVIDIA driver and CUDA toolkit installation")
        return False

    print("CUDA available: Yes")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Total memory: {props.total_memory / (1024**3):.1f} GB")
        print(f"  Multi-processor count: {props.multi_processor_count}")

    return True


def check_mps(torch):
    """Check Apple Metal (MPS) availability."""
    print("\n" + "=" * 60)
    print("APPLE METAL (MPS)")
    print("=" * 60)

    if platform.system() != "Darwin":
        print("MPS available: No (not macOS)")
        return False

    if not hasattr(torch.backends, 'mps'):
        print("MPS available: No (PyTorch version too old)")
        print("  Upgrade PyTorch: pip install --upgrade torch")
        return False

    mps_available = torch.backends.mps.is_available()
    mps_built = torch.backends.mps.is_built()

    print(f"MPS built in PyTorch: {mps_built}")
    print(f"MPS available: {mps_available}")

    if not mps_available:
        if not mps_built:
            print("  PyTorch was not built with MPS support")
            print("  Reinstall PyTorch: pip install --upgrade torch")
        else:
            print("  MPS is built but not available on this system")
            print("  Requires macOS 12.3+ and Apple Silicon or AMD GPU")
        return False

    # Test MPS functionality
    try:
        test_tensor = torch.zeros(1, device="mps")
        print("MPS test: Passed (tensor creation successful)")

        # Check for known issues
        print("\nMPS Compatibility Notes:")
        print("  - Use float32 dtype (float16 has limited support)")
        print("  - Flash Attention: Not supported")
        print("  - xFormers: Not supported")
        print("  - bitsandbytes quantization: Not supported")
        print("  - Expected performance: ~1.5-3x slower than RTX 4090")

        return True
    except Exception as e:
        print(f"MPS test: Failed ({e})")
        return False


def check_optional_libraries():
    """Check optional performance libraries."""
    print("\n" + "=" * 60)
    print("OPTIONAL LIBRARIES")
    print("=" * 60)

    # Flash Attention
    try:
        import flash_attn
        print(f"flash-attention: {flash_attn.__version__}")
    except ImportError:
        print("flash-attention: not installed")
        if platform.system() == "Darwin":
            print("  (Not supported on macOS)")

    # Triton
    try:
        import triton
        print(f"triton: {triton.__version__}")
    except ImportError:
        print("triton: not installed")
        if platform.system() == "Darwin":
            print("  (Not supported on macOS)")

    # SageAttention
    try:
        import sageattention
        version = getattr(sageattention, '__version__', 'installed (version unknown)')
        print(f"sageattention: {version}")
    except ImportError:
        print("sageattention: not installed")

    # xFormers
    try:
        import xformers
        version = getattr(xformers, '__version__', 'installed (version unknown)')
        print(f"xformers: {version}")
    except ImportError:
        print("xformers: not installed")
        if platform.system() == "Darwin":
            print("  (Limited support on macOS)")

    # Diffusers
    try:
        import diffusers
        print(f"diffusers: {diffusers.__version__}")
    except ImportError:
        print("diffusers: not installed")
        print("  Install with: pip install diffusers")

    # Transformers
    try:
        import transformers
        print(f"transformers: {transformers.__version__}")
    except ImportError:
        print("transformers: not installed")

    # Accelerate
    try:
        import accelerate
        print(f"accelerate: {accelerate.__version__}")
    except ImportError:
        print("accelerate: not installed")


def print_recommendations(has_cuda, has_mps):
    """Print device recommendations."""
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)

    if has_cuda:
        print("Recommended device: cuda")
        print("  - Full feature support")
        print("  - Best performance")
        print("\nconfig.yaml:")
        print("  model:")
        print("    device: cuda")
        print("    quantization: none  # or 8bit/4bit for lower VRAM")
    elif has_mps:
        print("Recommended device: mps")
        print("  - Native Apple Silicon acceleration")
        print("  - Good performance for local development")
        print("\nconfig.yaml:")
        print("  model:")
        print("    device: mps")
        print("    quantization: none  # quantization not supported on MPS")
    else:
        print("Recommended device: cpu")
        print("  WARNING: CPU inference is very slow!")
        print("  Consider using a machine with GPU support")
        print("\nconfig.yaml:")
        print("  model:")
        print("    device: cpu")
        print("    quantization: none")


def main():
    """Run all checks."""
    print("\n" + "=" * 60)
    print("  SCRATCHY ENVIRONMENT CHECK")
    print("=" * 60)

    get_system_info()

    torch = check_pytorch()
    if torch is None:
        print("\nCannot continue without PyTorch.")
        sys.exit(1)

    has_cuda = check_cuda(torch)
    has_mps = check_mps(torch)
    check_optional_libraries()
    print_recommendations(has_cuda, has_mps)

    print("\n" + "=" * 60)
    print("Check complete!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
