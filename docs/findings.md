# CUDA Compatibility Research Findings

**Date**: 2026-05-04  
**Branch**: `claude/research-cuda-compatibility-VW79l`  
**Scope**: Compared `docs/CUDA_COMPATIBILITY.md` against current state of the ecosystem.  
**Updated**: 2026-05-04 — second pass with nitpick verification (see [Corrections](#corrections-vs-first-pass) section).

---

## Summary

Several components documented in `CUDA_COMPATIBILITY.md` are out of date. Key changes: PyTorch 2.11.0 is out (CUDA 13 now the default), xFormers bumped to 0.0.35, SageAttention cu130 DLL crash resolved via woct0rdho post4 ABI3 wheel, Flash Attention 3 has Windows wheels, and `woct0rdho/triton-windows` is archived in favor of `triton-lang/triton-windows`. One significant erroneous claim from the first pass has been corrected: **FlashAttention-4 does NOT run on sm_120 (RTX 50-series consumer GPUs)**.

---

## Component-by-Component Findings

### 1. PyTorch

| | Docs state | Current state |
|-|-----------|---------------|
| Latest version | 2.10.x | **2.11.0** (released 2026-03-23) |
| torchvision | 0.24.1 | **0.26.0** (for 2.11) |
| torchaudio | 2.9.1 | **2.11.0** (for 2.11) |
| Supported CUDA | 12.6, 12.8, 13.0 | 12.6.3, 12.8.1, **12.9.1**, **13.0.2** |
| Default CUDA install | 12.8 recommended | **CUDA 13 is now the default** (x86_64 and ARM) |

Key changes in 2.11.0:
- **CUDA 13 is the new default**: `pip install torch` without `--index-url` now pulls cu130. Intentional cu128 installations must specify the index URL explicitly.
- **FlexAttention / FlashAttention-4 backend**: PyTorch 2.11 ships a FlashAttention-4 backend inside FlexAttention delivering 1.2×–3.2× speedups. **However, this only runs on SM90 (Hopper, H100) and SM100 (datacenter Blackwell, B200). Consumer RTX 50-series (SM120, including the 5070 Ti) do NOT benefit — see correction below.**
- No 2.11.1 patch release is planned (announced on PyTorch dev mailing list).
- CUDA 12.9 (cu129) wheel index exists but was a transitional slot, superseded by cu130.

Sources: [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/), [PyTorch GA announcement](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328), [No 2.11.1 announcement](https://dev-discuss.pytorch.org/t/no-pytorch-2-11-1-release/3352), [PyPI torch](https://pypi.org/project/torch/)

---

### 2. Flash Attention

| | Docs state | Current state |
|-|-----------|---------------|
| Latest FA2 version | 2.8.3 | **2.8.3** (unchanged) |
| FA2 cu130 / Python 3.12 Windows | ❌ (DLL fail) | **Partially resolved** — torch2.9.0 and torch2.10.0 wheels confirmed; torch2.11.0+cp312 **not confirmed** |
| FA3 Windows | Not mentioned | Available for cu128 and cu130 |
| FA4 on SM120 | Not mentioned | ❌ **Not supported** (SM100/datacenter-only) |

#### FA2 cu130 Windows — actual wheel availability (Python 3.12)

Confirmed wheels in Wildminder's repo as of May 2026:

| Wheel | Source | Status |
|-------|--------|--------|
| `flash_attn-2.8.3+cu130torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` | Wildminder | ✅ confirmed |
| `flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` | Wildminder | ✅ confirmed (date-stamped build) |
| `flash_attn-2.8.3+cu130torch2.11.0cxx11abiTRUE-cp312-cp312-win_amd64.whl` | — | ⚠️ **not confirmed** for cp312 |
| `flash_attn-2.8.3+cu130torch2.11.0cxx11abiTRUE-cp313-cp313-win_amd64.whl` | ussoewwin | ✅ confirmed (Python 3.13 only) |

Bottom line: if using Python 3.12 with PyTorch 2.11.0+cu130, a FA2 wheel may not exist yet. Use either PyTorch 2.10.0+cu130 (FA2 works), or Flash Attention 3.

#### Flash Attention 3

- Windows wheels are available via the [windreamer repository](https://windreamer.github.io/flash-attention3-wheels/), with cu130+torch2110, cu130+torch2100, cu130+torch291, cu130+torch290 combinations (last updated 2026-02-17).
- Official wheels also published by PyTorch at `download.pytorch.org/whl/flash-attn-3/`.
- FA3 requires SM90 (Hopper) or newer — **SM120 is newer than SM90 so FA3 is supported**, unlike FA4.
- CUDA 13.x MSVC source builds still require `/Zc:preprocessor`; `setup.py` does not inject it, so building from source on Windows will fail without a manual workaround.

Sources: [Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases), [windreamer FA3 wheels](https://windreamer.github.io/flash-attention3-wheels/), [ussoewwin HF repo](https://huggingface.co/ussoewwin/Flash-Attention-2_for_Windows/tree/main), [Wildminder HF repo](https://huggingface.co/Wildminder/AI-windows-whl/tree/main), [FA3 wheels PyTorch mailing list](https://dev-discuss.pytorch.org/t/flash-attention-3-wheels/3322)

---

### 3. SageAttention

| | Docs state | Current state |
|-|-----------|---------------|
| Latest version | 2.2.0 | **2.2.0** (last updated 2026-01-28, unchanged) |
| CUDA 13 / Windows | ❌ (DLL errors) | ✅ resolved — via woct0rdho **post4** ABI3 wheel specifically |
| SageAttention 3 | Not mentioned | Experimental (`sageattention3_blackwell` branch, unreleased) |

Key changes:
- The DLL crash on CUDA 13 Windows was NOT a general fix — it is specific to the woct0rdho **post4** ABI3 wheel. **Wildminder's `sageattention-2.2.0+cu130torch2.10.0-cp312` (post3) still crashes** with `DLL load failed while importing _fused` on import, as documented in [HF discussion #8](https://huggingface.co/Wildminder/AI-windows-whl/discussions/8).
- The fix is woct0rdho's post4 build, which switched to **libtorch stable ABI + ABI3**: `sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl`. The "torch2.9.0andhigher" name reflects that one wheel covers PyTorch ≥ 2.9 (including 2.10 and 2.11).
- A parallel cu128 ABI3 wheel exists: `sageattention-2.2.0+cu128torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl`.

Sources: [woct0rdho/SageAttention releases](https://github.com/woct0rdho/SageAttention/releases/tag/v2.2.0-windows.post4), [Wildminder DLL fail discussion](https://huggingface.co/Wildminder/AI-windows-whl/discussions/8), [Wan2GP cu130+2.10 confirmed working](https://github.com/deepbeepmeep/Wan2GP/issues/1480)

---

### 4. xFormers

| | Docs state | Current state |
|-|-----------|---------------|
| Latest version (cu128) | 0.0.33.post2 | — |
| Latest version (cu130) | 0.0.34 | **0.0.35** |
| Stable API/ABI | No mention | Migration complete: one xFormers build for PyTorch 2.10+ |

Key changes:
- **xFormers 0.0.35** is out, built against PyTorch 2.10.0. Thanks to PyTorch's stable C++ ABI migration, this version is compatible with any PyTorch ≥ 2.10 (including 2.11) without needing a separate build.
- Install command unchanged: `pip install xformers --index-url https://download.pytorch.org/whl/cu128` (or cu130).
- Some users have seen bare `pip install xformers` (without `--index-url`) silently pull a cu130 variant that upgrades PyTorch unexpectedly — always pin the index URL.
- V100 support was dropped in 0.0.30; minimum is SM80 (Ampere).

Sources: [facebookresearch/xformers releases](https://github.com/facebookresearch/xformers/releases), [PyPI xformers](https://pypi.org/project/xformers/)

---

### 5. Triton (Windows)

| | Docs state | Current state |
|-|-----------|---------------|
| Source repo | `woct0rdho/triton-windows` | **ARCHIVED** (2026-02-18, read-only) |
| Official home | Not mentioned | `https://github.com/triton-lang/triton-windows` |
| `pip install triton-windows` | Works | Still works (same PyPI package, same maintainers) |
| Triton version | 3.6.0 | **3.6.0** (same) |
| Triton 3.6 PyTorch requirement | Not stated | Requires **PyTorch >= 2.10** |

Key changes:
- `woct0rdho/triton-windows` was archived on 2026-02-18 and is now read-only. Development moved to the official **`triton-lang/triton-windows`** org (same maintainers: @woct0rdho, @jammm). `pip install triton-windows` is unaffected.
- Compatibility ladder: Triton 3.4 → PyTorch ≥ 2.8; Triton 3.5 → PyTorch ≥ 2.9; Triton 3.6 → PyTorch ≥ 2.10.

Sources: [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows), [woct0rdho/triton-windows (archived)](https://github.com/woct0rdho/triton-windows), [PyPI triton-windows](https://pypi.org/project/triton-windows/)

---

### 6. CUDA Toolkit

| | Docs state | Current state |
|-|-----------|---------------|
| Latest tested | 13.0 / 13.1 | **13.0.2** is current stable; cu129 (12.9) exists as transitional |
| Default in PyTorch 2.11 | N/A | **CUDA 13 is now the default** |
| sm_120 PTX support | CUDA 12.8+ | CUDA 12.9 adds PTX 8.7 for sm_120; 12.9 also adds sm_103/sm_121 (PTX 8.8) |

Key changes:
- **CUDA 12.9** (cu129) was released, introducing PTX 8.7 for sm_120 and PTX 8.8 for sm_103/sm_121 (newer Blackwell datacenter variants GB200/GB300). However cu129 was quickly superseded by cu130 and treated as a transitional index; most tooling targets cu130.
- RTX 50-series minimum driver remains 570+; CUDA 12.8+ is still the minimum for sm_120.

Sources: [NVIDIA CUDA 12.9 download archive](https://developer.nvidia.com/cuda-12-9-0-download-archive), [PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)

---

## Corrections vs First Pass

The following claims in the initial version of this document were **incorrect** and have been fixed:

### ❌ FA4 / FlexAttention does NOT benefit SM120 (RTX 5070 Ti)

**Wrong claim (first pass):**
> "Add PyTorch 2.11 FlexAttention / FA4 backend note under the RTX 50-series section — Blackwell users on PyTorch 2.11 can get better performance via FlexAttention without needing a separate flash-attn 3 install."

**Correction:**  
FlashAttention-4 is built around SM100 architectural features (TMEM, async tensor pipelines) that exist only in **datacenter Blackwell GPUs (B200, SM100)**. Consumer RTX 50-series GPUs are **SM120**, which is a different die that physically lacks these features. On SM120 hardware, frameworks fall back to Triton kernels or FA2/FA3 paths. This is confirmed by multiple sources including a dedicated Hardware Corner article and an open SGLang bug where frameworks incorrectly tried to invoke FA3/FA4 on SM120.

References: [Hardware Corner – RTX Pro 6000 Blackwell Does Not Support FA4](https://www.hardware-corner.net/rtx-pro-6000-blackwell-flashattention-4/), [SGLang bug #15342](https://github.com/sgl-project/sglang/issues/15342), [FA4 support for RTX 6000 Pro Blackwell issue](https://github.com/Dao-AILab/flash-attention/issues/2413), [FlashAttention SM120 issue](https://github.com/NVIDIA/Isaac-GR00T/issues/309)

### ❌ FA2 cu130torch2.11.0 for Python 3.12 was claimed without confirmation

**Wrong claim (first pass):**
> ussoewwin provides `flash_attn-2.8.3+cu130torch2.11.0cxx11abiTRUE-cp312-cp312-win_amd64.whl`

**Correction:**  
Verified wheel search results confirm ussoewwin only has **cp311 and cp313** (Python 3.11 and 3.13) for cu130+torch2.11.0. The cp312 (Python 3.12) variant for that exact combination has **not been found** in the public repositories. The ussoewwin repo has confirmed cu130+torch2.9.x/2.10.x wheels for cp312, and Wildminder confirms cu130+torch2.10.0 cp312. For cu130+torch2.11.0+cp312, either use Flash Attention 3, or downgrade PyTorch to 2.10.0.

### ❌ SageAttention "cu130 DLL fix" was attributed too broadly

**Wrong claim (first pass):**
> "woct0rdho now publishes wheels for cu130.torch2.11 and cu128.torch2.11 (win_amd64, ABI3). The CUDA 13 DLL issue that affected SageAttention 2.2.0 is resolved."

**Correction:**  
The Wildminder **post3** cu130 wheel still crashes (`DLL load failed while importing _fused`). The fix is specific to the woct0rdho **post4** ABI3 wheel. The wheel is named `sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl` — the "torch2.9.0andhigher" label indicates it covers PyTorch ≥ 2.9, not that there is a dedicated torch2.11 build.

---

## Quick Reference: Updated Compatibility Matrix

| PyTorch | torchvision | torchaudio | Supported CUDA | Notes |
|---------|-------------|-----------|---------------|-------|
| **2.11.0** | 0.26.0 | 2.11.0 | 12.6, 12.8, 12.9, **13.0** | Latest; CUDA 13 is now the pip default |
| 2.10.0 | 0.25.0 | 2.10.0 | 12.6, 12.8, 12.9, 13.0 | Recommended for full FA2+sage+xformers on Python 3.12 |
| 2.9.1 | 0.24.1 | 2.9.1 | 12.6, 12.8, 13.0 | Docs "recommended"; now superseded |
| 2.7.x | 0.22.x | 2.7.x | 12.4, 12.6, 12.8 | Minimum for Blackwell |

| Library | Docs version | Latest | CUDA 13 Windows + Python 3.12 | Notes |
|---------|-------------|--------|-------------------------------|-------|
| flash-attn 2 | 2.8.3 | 2.8.3 | ✅ up to torch2.10; ⚠️ torch2.11 unconfirmed | Use Wildminder cu130torch2.10 wheel |
| flash-attn 3 | — | available | ✅ | Windreamer or official PyTorch CDN |
| sageattention | 2.2.0 | 2.2.0 | ✅ post4 ABI3 only | Wildminder post3 still crashes |
| xformers | 0.0.33–0.0.34 | **0.0.35** | ✅ | Stable ABI; PyTorch 2.10+ compatible |
| triton-windows | 3.6.0 | 3.6.0 | ✅ | Repo moved to triton-lang org |

---

## Actionable Recommendations

1. **Update PyTorch** from `2.9.1+cu128` to `2.10.0+cu128` (stable, all libraries confirmed) or `2.10.0+cu130` (CUDA 13, FA2 cp312 wheel available). For CUDA 13 + Python 3.12, stop at PyTorch 2.10.0 for now — FA2 cp312 for torch2.11+cu130 is not yet confirmed.

2. **Add Flash Attention 3** as an alternative for CUDA 13 users (windreamer wheels; official PyTorch CDN). Clarify FA3 requires SM90+ (SM120 works); FA4 requires SM100+ (SM120 does **not** work).

3. **Correct the SageAttention CUDA 13 guidance**: recommend the woct0rdho **post4 ABI3** wheel specifically (`+cu130torch2.9.0andhigher.post4`), not the Wildminder cu130 build which still crashes.

4. **Bump xFormers** from `0.0.34` to `0.0.35`, and note the stable ABI change (one wheel works across PyTorch 2.10+).

5. **Update Triton link**: `woct0rdho/triton-windows` is archived; link to `triton-lang/triton-windows`. Note Triton 3.6 requires PyTorch ≥ 2.10.

6. **Add CUDA 12.9 row** to the toolkit table, and warn that bare `pip install torch` now defaults to cu130 as of PyTorch 2.11.

7. **Correct FA4 claim in the RTX 50-series section**: SM120 (5070 Ti, 5080, 5090) does **not** support FlashAttention-4. Only SM100 (datacenter B200) does. SM120 continues to use FA2 or FA3 paths.
