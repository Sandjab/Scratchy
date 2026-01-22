# CUDA / PyTorch Compatibility Guide

This guide documents version compatibility requirements for running Scratchy with GPU acceleration.

## Quick Reference

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| NVIDIA Driver | 525+ | 550+ |
| CUDA Toolkit | 12.1 | 12.8 |
| PyTorch | 2.2+ | 2.9+ |
| Python | 3.9 | 3.10-3.11 |

## PyTorch / CUDA Compatibility Matrix

| PyTorch | Supported CUDA Versions |
|---------|------------------------|
| 2.9.x | 12.6, 12.8, 13.0 |
| 2.7.x | 12.4, 12.6 |
| 2.5.x | 11.8, 12.1, 12.4 |
| 2.4.x | 11.8, 12.1, 12.4 |
| 2.2.x | 11.8, 12.1 |

**Important**: The CUDA version shown in `nvidia-smi` is the *driver's maximum supported CUDA version*, not the installed toolkit version. Your driver must support a CUDA version >= the toolkit you install.

## Flash Attention Requirements

Flash Attention provides significant memory and speed improvements but has strict requirements:

| Requirement | Flash Attention 2 | Flash Attention 3 |
|-------------|-------------------|-------------------|
| GPU Architecture | Ampere+ (SM80) | Hopper (H100/H800) |
| CUDA | 12.0+ | 12.3+ (12.8 recommended) |
| PyTorch | 2.2+ | 2.2+ |
| Supported GPUs | A100, RTX 3090, RTX 4090, RTX 5070+ | H100, H800 |
| Data Types | fp16, bf16 | fp16, bf16 |

**Not supported**: Turing GPUs (T4, RTX 2080) - use Flash Attention 1.x

### Head Dimension Limits
- All head dimensions up to 256 supported
- Head dim > 192 backward pass requires A100/H100
- Head dim 256 backward works on consumer GPUs (no dropout) as of flash-attn 2.5.5

## xFormers Requirements

| Requirement | Value |
|-------------|-------|
| Platform | Linux (primary), Windows (CUDA 12.4+) |
| Python | 3.9 - 3.13 |
| CUDA (Flash-Attention 3) | 12.3+ |

## Diffusers Library

Diffusers works with PyTorch 2.0+ and benefits significantly from modern CUDA architectures:

| GPU | Speedup with PyTorch 2.0+ |
|-----|--------------------------|
| A100 | ~50% |
| RTX 4090 | 35-50% |
| RTX 3090 | 20-35% |

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

# For CUDA 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 3. Flash Attention on Unsupported GPU
**Symptom**: `RuntimeError: FlashAttention only supports Ampere GPUs or newer`

**Solution**: Disable flash attention or use xFormers memory-efficient attention:
```python
pipe.enable_xformers_memory_efficient_attention()
# or
pipe.enable_attention_slicing()
```

### 4. bf16 on Pre-Ampere GPUs
**Symptom**: `RuntimeError: "LayerNormKernelImpl" not implemented for 'BFloat16'`

**Solution**: Use fp16 instead of bf16 on Turing and older GPUs:
```python
pipe = pipeline.from_pretrained(model_id, torch_dtype=torch.float16)
```

### 5. triton Not Available on Windows
**Symptom**: `ModuleNotFoundError: No module named 'triton'`

**Solution**: Triton has limited Windows support. Use WSL2 or disable triton-dependent features.

### 6. CUDA Out of Memory with Large Models
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

Expected output:
```
python version: 3.10.x
torch version: 2.x.x
cuda version (torch): 12.x
cuda available: True
flash-attention version: 2.x.x (or "not installed")
xformers version: 0.x.x (or "not installed")
```

## Recommended Installation

For best compatibility with Scratchy on Windows with RTX 5070 Ti:

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate

# Install PyTorch with CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install Scratchy
pip install -e .

# Optional: Install xformers (if available for your CUDA version)
pip install xformers
```

## Resources

- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [PyTorch-CUDA Compatibility Matrix](https://github.com/eminsafa/pytorch-cuda-compatibility)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [xFormers Platform Compatibility](https://deepwiki.com/facebookresearch/xformers/3.1-platform-compatibility)
- [Diffusers Installation](https://huggingface.co/docs/diffusers/en/installation)
