# Model Configuration Guide

This guide covers how to configure and use different model sources in Scratchy.

## Table of Contents

1. [Model Sources Overview](#model-sources-overview)
2. [Using Local Models](#using-local-models)
3. [Using CivitAI Models](#using-civitai-models)
4. [Using Direct URLs](#using-direct-urls)
5. [Model Format Details](#model-format-details)
6. [Storage and Cache Management](#storage-and-cache-management)
7. [Troubleshooting](#troubleshooting)
8. [Examples](#examples)

---

## Model Sources Overview

Scratchy supports loading models from multiple sources:

| Source | Pros | Cons | Best For |
|--------|------|------|----------|
| **HuggingFace** (built-in) | Easy setup, automatic downloads | Limited to official models | Quick start, production |
| **Local Path** | Full control, offline use | Manual management | Custom/fine-tuned models |
| **CivitAI** | Huge model library, community models | Requires download | Exploring new models |
| **Direct URL** | Flexible, any source | No metadata | Specific model files |

### Priority Order

When loading a custom model, sources are checked in this order:
1. `local_path` - Use local file if specified
2. `civitai_model_id` - Download from CivitAI if specified
3. `download_url` - Download from URL if specified
4. Fall back to HuggingFace model ID

---

## Using Local Models

Point Scratchy to existing model files on your system.

### Supported Formats

- **Single file checkpoints**: `.safetensors`, `.ckpt`, `.pt`, `.bin`
- **Diffusers format**: Directory with `model_index.json` and component folders

### Configuration

```yaml
model:
  name: "custom"
  local_path: "D:/models/my_model.safetensors"
  pipeline_type: "sdxl"  # Optional: auto-detected
```

### Path Formats

**Windows:**
```yaml
local_path: "D:/models/my_model.safetensors"
local_path: "D:\\models\\my_model.safetensors"
```

**Linux/Mac:**
```yaml
local_path: "/home/user/models/my_model.safetensors"
local_path: "~/models/my_model.safetensors"
```

**Relative paths:**
```yaml
local_path: "./models/my_model.safetensors"
```

### Directory Structure

For single file checkpoints:
```
D:/models/
  my_model.safetensors
```

For diffusers format:
```
D:/models/my_model/
  model_index.json
  scheduler/
  text_encoder/
  text_encoder_2/
  tokenizer/
  tokenizer_2/
  unet/
  vae/
```

---

## Using CivitAI Models

Download models directly from [CivitAI](https://civitai.com/).

### CLI Download (Recommended)

```bash
# Download by URL
scratchy-models download https://civitai.com/models/12345

# Download by model ID
scratchy-models download 12345

# Specify version
scratchy-models download 12345 --version 67890

# List downloaded models
scratchy-models list

# Show model info
scratchy-models info "Model Name"
```

### Config-Based Auto-Download

Models are automatically downloaded on first server start:

```yaml
model:
  name: "custom"
  civitai_model_id: "12345"
  civitai_version_id: "67890"  # Optional
```

### Finding Models on CivitAI

1. Go to [civitai.com](https://civitai.com/)
2. Browse or search for a model
3. The URL contains the model ID: `civitai.com/models/12345`
4. For specific versions, click the version and note the `modelVersionId` parameter

### Understanding Model Types

| Type | Description | Scratchy Support |
|------|-------------|------------------|
| Checkpoint | Full model file | Yes |
| Safetensors | Optimized format | Yes (preferred) |
| LoRA | Fine-tuning weights | Not yet |
| Textual Inversion | Embeddings | Not yet |

### API Key (Optional)

For faster downloads, set a CivitAI API key:

```bash
export CIVITAI_API_KEY="your_api_key_here"
```

Get your API key from [CivitAI Account Settings](https://civitai.com/user/account).

### Trigger Words

Some CivitAI models require specific trigger words in prompts. After downloading, check the model info:

```bash
scratchy-models info "Model Name"
```

The trigger words will be displayed and should be included in your prompts.

---

## Using Direct URLs

Download models from any accessible URL.

### Configuration

```yaml
model:
  name: "custom"
  download_url: "https://example.com/model.safetensors"
```

### CLI Download

```bash
# Auto-detect filename
scratchy-models download https://example.com/model.safetensors

# Specify filename
scratchy-models download https://example.com/model.safetensors --name my_model.safetensors
```

### Supported URL Types

- Direct file URLs: `https://example.com/model.safetensors`
- GitHub releases: `https://github.com/user/repo/releases/download/v1.0/model.safetensors`
- Cloud storage (with direct links):
  - Google Drive: Use "Get Link" and ensure it's a direct download link
  - Dropbox: Change `?dl=0` to `?dl=1` for direct download

### Security Considerations

- Only download from trusted sources
- Verify file hashes when available
- Safetensors format is safer than pickle-based formats (`.ckpt`, `.pt`)

---

## Model Format Details

### Single File vs Diffusers Format

| Format | File Types | Loading Method | Pros | Cons |
|--------|------------|----------------|------|------|
| Single File | `.safetensors`, `.ckpt` | `from_single_file()` | Compact, portable | Slower first load |
| Diffusers | Directory with components | `from_pretrained()` | Faster loading | More disk space |

### Pipeline Type Detection

Scratchy auto-detects the pipeline type from:
1. CivitAI metadata (`base_model` field)
2. `model_index.json` in diffusers format
3. Filename patterns (e.g., "xl", "sdxl", "flux")

Override with explicit configuration:
```yaml
model:
  name: "custom"
  local_path: "./model.safetensors"
  pipeline_type: "sdxl"  # flux, sdxl, sd15, or auto
```

### Supported Pipelines

| Pipeline | Description | Config Value |
|----------|-------------|--------------|
| FLUX | Black Forest Labs FLUX models | `flux` |
| SDXL | Stable Diffusion XL | `sdxl` |
| SD 1.5 | Stable Diffusion 1.5 | `sd15` |
| Auto | Let Scratchy detect | `auto` (default) |

---

## Storage and Cache Management

### Default Storage Location

Downloaded models are stored in:
```
./scratchy_data/models/
  civitai/           # CivitAI downloads
    12345_67890/     # model_id_version_id
      model.safetensors
      metadata.json
  url/               # URL downloads
    a1b2c3d4e5f6/    # URL hash
      model.safetensors
      metadata.json
```

### Configuring Storage

```yaml
storage:
  models_dir: "./scratchy_data/models"  # Change download location
```

### Disk Space Requirements

| Model Type | Approximate Size |
|------------|------------------|
| SD 1.5 | 2-4 GB |
| SDXL | 6-8 GB |
| FLUX.1-schnell | 10-12 GB |
| FLUX.1-dev | 14-16 GB |

### Managing Downloaded Models

```bash
# List all models with sizes
scratchy-models list

# Remove a model
scratchy-models remove "Model Name"
scratchy-models remove "Model Name" --yes  # Skip confirmation
```

---

## Troubleshooting

### Common Errors

**"Model path not found"**
- Check the path exists and is accessible
- Verify file permissions
- Use absolute paths or paths relative to working directory

**"Could not detect pipeline type"**
- Set `pipeline_type` explicitly in config
- Check if the model file is corrupted

**"Download failed"**
- Check internet connection
- Verify the URL is accessible
- For CivitAI, ensure the model isn't deleted or restricted

**"CUDA out of memory"**
- Enable quantization: `quantization: "8bit"` or `quantization: "4bit"`
- Use a smaller model
- Close other GPU applications

**"Hash mismatch"**
- Re-download the model
- The file may be corrupted or modified

### VRAM Issues

If you run out of VRAM:

1. **Enable quantization:**
   ```yaml
   model:
     quantization: "8bit"  # ~50% VRAM reduction
   ```

2. **Use a smaller model:**
   - SDXL: ~8 GB VRAM
   - SD 1.5: ~4 GB VRAM

3. **Enable memory optimizations** (automatic):
   - Attention slicing
   - VAE slicing

### Download Failures

Downloads support automatic resume. If a download fails:

1. Run the same command again - it will resume from where it stopped
2. Partial downloads are stored with `.partial` extension
3. To force re-download, remove the partial file manually

---

## Examples

### Example 1: CivitAI Model with CLI

```bash
# Download a popular SDXL model
scratchy-models download https://civitai.com/models/139562

# Check the download
scratchy-models list

# Get config suggestion
scratchy-models info "Juggernaut XL"
```

Then use in config:
```yaml
model:
  name: "custom"
  local_path: "./scratchy_data/models/civitai/139562_274039/juggernautXL.safetensors"
```

### Example 2: Auto-Download on Server Start

```yaml
model:
  name: "custom"
  civitai_model_id: "139562"
  # Model downloads automatically on first `python -m scratchy.main`
```

### Example 3: Local Model Collection

```yaml
model:
  name: "custom"
  local_path: "D:/AI/models/realisticVision_v60.safetensors"
  pipeline_type: "sd15"
```

### Example 4: Direct URL from GitHub

```yaml
model:
  name: "custom"
  download_url: "https://github.com/user/repo/releases/download/v1.0/model.safetensors"
```

### Example 5: Environment Variable Override

```bash
# Set model path via environment variable
export SCRATCHY_MODEL__NAME=custom
export SCRATCHY_MODEL__LOCAL_PATH=/models/my_model.safetensors
export SCRATCHY_MODEL__PIPELINE_TYPE=sdxl

python -m scratchy.main
```

---

## Migration from HuggingFace-Only Setup

If you're currently using built-in models and want to switch to a custom model:

**Before (HuggingFace):**
```yaml
model:
  name: "sdxl"
```

**After (Custom):**
```yaml
model:
  name: "custom"
  civitai_model_id: "12345"
  # Or:
  # local_path: "./my_model.safetensors"
```

The server will download and use the new model on restart. Your existing API keys, credits, and settings remain unchanged.
