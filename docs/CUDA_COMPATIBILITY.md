# CUDA / PyTorch Compatibility Guide

This guide documents version compatibility requirements for running Scratchy with GPU acceleration.

## Quick Reference

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| NVIDIA Driver | 525+ | 580+ (for RTX 50-series) |
| CUDA Toolkit | 12.1 | 12.8 |
| PyTorch | 2.2+ | 2.9.1+ |
| Python | 3.9 | 3.12 |

## Verified Working Configuration (RTX 5070 Ti)

Tested on Windows 11 with NVIDIA RTX 5070 Ti (16GB VRAM, sm_120 Blackwell):

| Component | Version |
|-----------|---------|
| Python | 3.12.10 |
| PyTorch | 2.9.1+cu128 |
| CUDA (torch) | 12.8 |
| torchvision | 0.24.1+cu128 |
| torchaudio | 2.9.1+cu128 |
| flash-attention | 2.8.3 |
| triton | 3.6.0 |
| sageattention | 2.2.0 |
| xformers | 0.0.33.post2 |
| Driver | 580.97 |
| CUDA Toolkit | 12.8 |

## CUDA 13.0/13.1 Testing Results (RTX 5070 Ti)

Tested on Windows 11 with NVIDIA RTX 5070 Ti, Driver 580.97, CUDA Toolkit 13.0 and 13.1:

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.12.10 | ✅ Working |
| PyTorch | 2.10.0+cu130 | ✅ Working |
| torchvision | 0.25.0+cu130 | ✅ Working |
| torchaudio | 2.10.0+cu130 | ✅ Working |
| CUDA available | Yes | ✅ Working |
| triton | 3.6.0 | ✅ Working |
| xformers | 0.0.34 | ✅ Working |
| flash-attention | 2.8.3 | ❌ DLL load failed |
| sageattention | 2.2.0 | ❌ DLL load failed |

### Issue: Prebuilt Wheels Incompatible

The prebuilt Windows wheels from [Wildminder](https://huggingface.co/Wildminder/AI-windows-whl) for flash-attention and sageattention (cu130/torch2.10.0) fail with:

```
ImportError: DLL load failed while importing flash_attn_2_cuda: The specified procedure was not found.
```

#### Root Cause: PyTorch C++ ABI Mismatch

Tested with both `CUDA_PATH` and `PATH` correctly set to CUDA 13.0 - **same error persists**. This confirms the issue is **not** the CUDA toolkit version, but a C++ ABI mismatch.

**Why CUDA environment variables don't help:**

| Variable | Purpose | Relevance |
|----------|---------|-----------|
| `PATH` | Windows DLL loader finds CUDA runtime DLLs | Runtime loading |
| `CUDA_PATH` | Apps/build tools locate CUDA installation | Building from source |

PyTorch 2.10.0+cu130 **bundles its own CUDA libraries** internally. The flash-attention wheel was compiled against a different PyTorch build (likely a nightly or pre-release) with different internal C++ symbols. When the wheel tries to link against the official PyTorch release, the symbols don't match.

**Keep PATH and CUDA_PATH aligned** for consistency:
```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin;" + $env:PATH
```

### Recommendation

**Use CUDA 12.8 with PyTorch 2.9.1** for full compatibility:
- All attention libraries work (flash-attention, sageattention, xformers)
- Stable, tested configuration
- Keep `.venv130` for future testing when updated wheels become available

### Alternative: CUDA 13.0 with xformers Only

If you need PyTorch 2.10.0 features, the cu130 environment works with xformers (which provides `memory_efficient_attention`):

```bash
py -3.12 -m venv .venv130
.venv130\Scripts\activate
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
pip install triton-windows
pip install xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu130
pip install -e .
```

## RTX 50-Series (Blackwell) Specific Notes

The RTX 5070, 5070 Ti, 5080, and 5090 use the **Blackwell architecture** with compute capability **sm_120**.

### Key Requirements
- **CUDA Toolkit**: 12.8 or higher required for sm_120 support
- **PyTorch**: 2.7+ with cu128 or higher
- **Driver**: 570+ (open-source nvidia-driver-570-open recommended on Linux)

### Python Version Compatibility

| Python | PyTorch cu128 | flash-attention | xformers | sageattention |
|--------|---------------|-----------------|----------|---------------|
| 3.14 | Yes | **NO** | **NO** | Yes (ABI3) |
| 3.13 | Yes | Yes | Yes (ABI3) | Yes (ABI3) |
| 3.12 | Yes | Yes | Yes | Yes |
| 3.11 | Yes | Yes | Yes | Yes |
| 3.10 | Yes | Limited | Yes | Yes |

**Recommendation**: Use **Python 3.12** for full ecosystem support on Windows.

## PyTorch / CUDA Compatibility Matrix

| PyTorch | Supported CUDA Versions | Notes |
|---------|------------------------|-------|
| 2.10.x | 12.8, 13.0 | Latest, xformers 0.0.34 requires this |
| 2.9.x | 12.6, 12.8, 13.0 | Recommended for RTX 50-series |
| 2.7.x | 12.4, 12.6, 12.8 | Minimum for Blackwell |
| 2.5.x | 11.8, 12.1, 12.4 | Pre-Blackwell only |

**Important**:
- The CUDA version in `nvidia-smi` is the *driver's maximum supported CUDA version*, not the installed toolkit
- Match your PyTorch CUDA version (cu128/cu130) to your installed CUDA Toolkit to avoid DLL loading errors
- Using cu130 PyTorch with CUDA 12.8 toolkit may cause `DLL load failed` errors for flash-attention/sageattention

## Flash Attention Requirements

Flash Attention provides significant memory and speed improvements but has strict requirements:

| Requirement | Flash Attention 2 | Flash Attention 3 |
|-------------|-------------------|-------------------|
| GPU Architecture | Ampere+ (SM80) | Hopper (H100/H800) |
| CUDA | 12.0+ | 12.3+ (12.8 recommended) |
| PyTorch | 2.2+ | 2.2+ |
| Supported GPUs | A100, RTX 3090, RTX 4090, **RTX 5070+** | H100, H800 |
| Data Types | fp16, bf16 | fp16, bf16 |

**Not supported**: Turing GPUs (T4, RTX 2080) - use Flash Attention 1.x

### Head Dimension Limits
- All head dimensions up to 256 supported
- Head dim > 192 backward pass requires A100/H100
- Head dim 256 backward works on consumer GPUs (no dropout) as of flash-attn 2.5.5

## SageAttention Requirements

SageAttention 2.2+ provides 2-5x speedup over FlashAttention with quantized attention:

| Requirement | SageAttention 2.2 |
|-------------|-------------------|
| GPU Architecture | SM89 (RTX 40xx) or SM120 (RTX 50xx) for SageAttention2++ |
| CUDA | 12.8+ |
| PyTorch | 2.7+ |
| Supported GPUs | RTX 40xx, **RTX 50xx (Blackwell)** |

## xFormers Requirements

| Requirement | Value |
|-------------|-------|
| Platform | Linux (primary), Windows (CUDA 12.4+) |
| Python | 3.9 - 3.13 (ABI3 wheels) |
| CUDA (Flash-Attention 3) | 12.3+ |

## Pre-built Windows Wheels

Since many AI packages are difficult to compile on Windows, use these pre-built wheel sources:

### Flash Attention
- **Wildminder**: https://huggingface.co/Wildminder/AI-windows-whl
- **ussoewwin**: https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows
- **lldacing**: https://huggingface.co/lldacing/flash-attention-windows-wheel

### SageAttention
- **woct0rdho** (recommended): https://github.com/woct0rdho/SageAttention/releases
- **sdbds**: https://github.com/sdbds/SageAttention-for-windows/releases
- **Wildminder**: https://huggingface.co/Wildminder/AI-windows-whl

### xFormers
- **PyTorch Official**: `pip install xformers --index-url https://download.pytorch.org/whl/cu128`
- **Wildminder**: https://huggingface.co/Wildminder/AI-windows-whl

### Triton
- **triton-windows**: `pip install triton-windows` (includes CUDA toolchain since v3.2.0)
- **woct0rdho**: https://github.com/woct0rdho/triton-windows

## Common Incompatibilities

### 1. CUDA Toolkit vs Driver Mismatch
**Symptom**: `CUDA driver version is insufficient for CUDA runtime version`

**Solution**: Update your NVIDIA driver or install an older CUDA toolkit version supported by your driver.

### 2. PyTorch Built for Wrong CUDA Version
**Symptom**: `torch.cuda.is_available()` returns `False` despite having a GPU

**Solution**: Install PyTorch with the correct CUDA version:
```bash
# For CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# For CUDA 13.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### 3. DLL Load Failed for flash-attention/sageattention
**Symptom**: `ImportError: DLL load failed while importing flash_attn_2_cuda: The specified procedure was not found`

**Causes**:
1. PyTorch CUDA version doesn't match installed CUDA Toolkit
2. Prebuilt wheel was compiled against a different PyTorch build (ABI mismatch)

**Solutions**:

For CUDA toolkit mismatch - use matching versions:
```bash
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install <wheel-url-matching-cu128-and-torch-version>
```

For CUDA 13.0/13.1 with PyTorch 2.10.0 - the current Wildminder wheels have ABI issues. Use xformers instead:
```bash
pip install xformers==0.0.34 --index-url https://download.pytorch.org/whl/cu130
# Skip flash-attention and sageattention until updated wheels are available
```

### 4. Python 3.14 Missing Wheels
**Symptom**: `No matching distribution found for flash-attn` on Python 3.14

**Cause**: flash-attention and xformers don't have Python 3.14 wheels yet

**Solution**: Use Python 3.12 for full compatibility

### 5. Flash Attention on Unsupported GPU
**Symptom**: `RuntimeError: FlashAttention only supports Ampere GPUs or newer`

**Solution**: Disable flash attention or use xFormers memory-efficient attention:
```python
pipe.enable_xformers_memory_efficient_attention()
# or
pipe.enable_attention_slicing()
```

### 6. bf16 on Pre-Ampere GPUs
**Symptom**: `RuntimeError: "LayerNormKernelImpl" not implemented for 'BFloat16'`

**Solution**: Use fp16 instead of bf16 on Turing and older GPUs:
```python
pipe = pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

### 7. xFormers Version Mismatch
**Symptom**: `WARNING[XFORMERS]: xFormers can't load C++/CUDA extensions`

**Cause**: xformers was built for a different PyTorch version

**Solution**: Install xformers from the matching PyTorch wheel index:
```bash
pip install xformers --index-url https://download.pytorch.org/whl/cu128
```

### 8. CUDA Out of Memory with Large Models
**Symptom**: `CUDA out of memory`

**Solutions**:
- Enable attention slicing: `pipe.enable_attention_slicing()`
- Enable VAE slicing: `pipe.vae.enable_slicing()`
- Use 8-bit quantization in config
- Use sequential CPU offload: `pipe.enable_sequential_cpu_offload()`

## Checking Your Setup

Run the included `check.py` script to verify your environment:

```bash
python check.py
```

Expected output (RTX 5070 Ti example):
```
python version: 3.12.10
torch version: 2.9.1+cu128
cuda version (torch): 12.8
torchvision version: 0.24.1+cu128
torchaudio version: 2.9.1+cu128
cuda available: True
flash-attention version: 2.8.3
triton version: 3.6.0
sageattention is installed but has no __version__ attribute
xformers: 0.0.33.post2
```

## Recommended Installation (RTX 5070 Ti / Windows)

```bash
# Install Python 3.12 if needed
winget install Python.Python.3.12

# Create virtual environment
py -3.12 -m venv .venv
.venv\Scripts\activate

# Install PyTorch with CUDA 12.8
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Install triton-windows
pip install triton-windows

# Install SageAttention (pick wheel matching your torch version)
pip install https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post3/sageattention-2.2.0+cu128torch2.9.0.post3-cp39-abi3-win_amd64.whl

# Install flash-attention (pick wheel matching your torch version and Python)
pip install https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/flash_attn-2.8.3+cu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl

# Install xformers from official index
pip install xformers==0.0.33.post2 --index-url https://download.pytorch.org/whl/cu128

# Install Scratchy
pip install -e .
```

## Resources

- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch-CUDA Compatibility Matrix](https://github.com/eminsafa/pytorch-cuda-compatibility)
- [PyTorch sm_120 Blackwell Support Issue](https://github.com/pytorch/pytorch/issues/164342)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [SageAttention GitHub](https://github.com/thu-ml/SageAttention)
- [SageAttention Windows Releases](https://github.com/woct0rdho/SageAttention/releases)
- [triton-windows PyPI](https://pypi.org/project/triton-windows/)
- [xFormers Platform Compatibility](https://deepwiki.com/facebookresearch/xformers/3.1-platform-compatibility)
- [Diffusers Installation](https://huggingface.co/docs/diffusers/en/installation)
- [Wildminder AI Windows Wheels](https://github.com/wildminder/AI-windows-whl)
