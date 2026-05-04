# CUDA Compatibility Research Findings

**Date**: 2026-05-04
**Branch**: `claude/research-cuda-compatibility-VW79l`
**Scope**: Compared `docs/CUDA_COMPATIBILITY.md` against current state of the ecosystem.

---

## Summary

Several components documented in `CUDA_COMPATIBILITY.md` are now out of date. The most significant changes are: PyTorch 2.11.0 is out (with CUDA 13 as the new default), Flash Attention 3 has official Windows wheels, SageAttention wheels now work with CUDA 13 / PyTorch 2.11, xFormers bumped to 0.0.35, and the `woct0rdho/triton-windows` repository has been archived in favor of the official `triton-lang/triton-windows`.

---

## Component-by-Component Findings

### 1. PyTorch

| | Docs state | Current state |
|-|-----------|---------------|
| Latest version | 2.10.x | **2.11.0** (released 2026-03-23) |
| Supported CUDA | 12.6, 12.8, 13.0 | 12.6.3, 12.8.1, **12.9.1**, **13.0.2** |
| Default CUDA | 12.8 recommended | **CUDA 13 is now the default** for x86_64 and ARM installs |

Key changes in 2.11.0:
- **FlexAttention** now has a FlashAttention-4 backend on Hopper and Blackwell GPUs, delivering 1.2×–3.2× speedups over the existing Triton path on compute-bound workloads.
- No 2.11.1 patch release is planned (announced on PyTorch dev mailing list).
- CUDA 12.9 (cu129) wheel index exists but was treated as a transitional slot and superseded by cu130.

Sources: [PyTorch 2.11 Release Blog](https://pytorch.org/blog/pytorch-2-11-release-blog/), [PyTorch GA announcement](https://dev-discuss.pytorch.org/t/pytorch-2-11-0-general-availability/3328), [PyPI](https://pypi.org/project/torch/)

---

### 2. Flash Attention

| | Docs state | Current state |
|-|-----------|---------------|
| Latest FA2 version | 2.8.3 | **2.8.3** (unchanged) |
| CUDA 13 / Windows DLL issue | Still broken | **Flash Attention 3** now has official Windows wheels with cu130 support |
| FA3 Windows availability | Not mentioned | Available via [windreamer's repo](https://windreamer.github.io/flash-attention3-wheels/) and officially at `download.pytorch.org/whl/flash-attn-3/` |

Key changes:
- **Flash Attention 3** official Windows wheels were added to PyTorch's CDN (`download.pytorch.org/whl/flash-attn-3/`), covering CUDA 12.6+, 13.x, x86/ARM, Linux/Windows.
- CUDA 13.0 Windows support for FA3 was confirmed working as of 2026-03-19.
- FA3 requires **SM90 (Hopper) or newer** — it is supported on Blackwell (sm_120) as it is "newer".
- The DLL ABI mismatch for FA2 prebuilt wheels on CUDA 13 (`ImportError: DLL load failed`) remains unresolved for FA2. FA3 wheels are the recommended path for CUDA 13 on Windows.
- CUDA 13.x MSVC compilation requires `/Zc:preprocessor` (standard conforming preprocessor); `setup.py` does not pass this flag automatically, so source builds still fail without a workaround.
- Community wheels: [IxaOne (HF, FA3 Blackwell/sm_120, Python 3.13)](https://huggingface.co/IxaOne/flash-attn-blackwell-win-cp313), [mjun0812 prebuild wheels (Linux + Windows, FA2 + FA3)](https://github.com/mjun0812/flash-attention-prebuild-wheels).

Sources: [Dao-AILab/flash-attention releases](https://github.com/Dao-AILab/flash-attention/releases), [windreamer FA3 wheels](https://windreamer.github.io/flash-attention3-wheels/), [PyTorch dev mailing list – FA3 wheels](https://dev-discuss.pytorch.org/t/flash-attention-3-wheels/3322)

---

### 3. SageAttention

| | Docs state | Current state |
|-|-----------|---------------|
| Latest version | 2.2.0 | **2.2.0** (last updated 2026-01-28, unchanged) |
| CUDA 13 / PyTorch 2.10-2.11 wheels | Not available (DLL errors) | **Now available** from woct0rdho |
| SageAttention 3 | Not mentioned | Experimental (`sageattention3_blackwell` branch, not yet released as stable) |

Key changes:
- The woct0rdho fork now publishes wheels for **cu130.torch2.11** and **cu128.torch2.11** (win_amd64, ABI3). The CUDA 13 DLL issue that affected SageAttention 2.2.0 is resolved with these newer wheel builds.
- The ABI3 wheel is confirmed working with PyTorch 2.10 + CUDA 13 on Windows 11 (reported in [Wan2GP issue #1480](https://github.com/deepbeepmeep/Wan2GP/issues/1480)).
- SageAttention3 (`sageattention3_blackwell` branch) targets Blackwell sm_120 specifically; Windows wheel availability is still in development.

Sources: [woct0rdho/SageAttention releases](https://github.com/woct0rdho/SageAttention/releases), [thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)

---

### 4. xFormers

| | Docs state | Current state |
|-|-----------|---------------|
| Latest version (cu128) | 0.0.33.post2 | — |
| Latest version (cu130) | 0.0.34 | **0.0.35** |
| Stable API/ABI | No mention | Migration complete: xFormers 0.0.35 targets PyTorch 2.10+ stable ABI, compatible with any later PyTorch |

Key changes:
- **xFormers 0.0.35** is out, built for PyTorch 2.10.0. Thanks to the PyTorch stable API/ABI migration, this version is compatible with PyTorch 2.10 and any later release (including 2.11).
- The install command for cu128 / cu130 is unchanged: `pip install xformers --index-url https://download.pytorch.org/whl/cu128` (or cu130).
- Some users have seen dependency resolution pull 2.11.0+cu130 when resolving from PyPI without a pinned index, which can be unexpected.
- V100 support was dropped in 0.0.30; the minimum is now SM80 (Ampere).

Sources: [facebookresearch/xformers releases](https://github.com/facebookresearch/xformers/releases), [PyPI xformers](https://pypi.org/project/xformers/)

---

### 5. Triton (Windows)

| | Docs state | Current state |
|-|-----------|---------------|
| Source repo | `woct0rdho/triton-windows` | **ARCHIVED** (2026-02-18, read-only) |
| Official home | Not mentioned | `https://github.com/triton-lang/triton-windows` |
| `pip install triton-windows` | Works | Still works (same package name on PyPI) |
| Triton version | 3.6.0 | **3.6.0** (same) |
| Triton 3.6 PyTorch requirement | Not stated | Requires **PyTorch >= 2.10** |

Key changes:
- `woct0rdho/triton-windows` was archived on 2026-02-18. Development moved to the official **`triton-lang/triton-windows`** repository (same maintainers: @woct0rdho, @jammm).
- The `triton-windows` PyPI package continues to receive updates. `pip install triton-windows` is unaffected.
- Version/PyTorch compatibility ladder: Triton 3.4 → PyTorch ≥ 2.8; Triton 3.5 → PyTorch ≥ 2.9; Triton 3.6 → PyTorch ≥ 2.10.

Sources: [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows), [woct0rdho/triton-windows](https://github.com/woct0rdho/triton-windows), [PyPI triton-windows](https://pypi.org/project/triton-windows/)

---

### 6. CUDA Toolkit

| | Docs state | Current state |
|-|-----------|---------------|
| Latest toolkit tested | 13.0 / 13.1 | **12.9** also exists (cu129); **13.0.2** is latest stable |
| Default in PyTorch 2.11 | Not applicable | **CUDA 13 is now the default** (x86_64 and ARM) |
| sm_120 PTX support | CUDA 12.8+ | CUDA 12.9 adds PTX 8.7 for sm_120; CUDA 12.9 also adds sm_103/sm_121 (PTX 8.8) |

Key changes:
- **CUDA 12.9** was released and introduced PTX 8.7 (sm_120) and PTX 8.8 (sm_103, sm_121, used in newer Blackwell variants such as GB200/GB300). There is a cu129 wheel index at `download.pytorch.org/whl/cu129` but it was a transitional slot; most toolchains have moved to cu130.
- With PyTorch 2.11, **CUDA 13.0 is now the default install target**, not 12.8. Users who `pip install torch` without a specific `--index-url` will get cu130 builds.
- RTX 50-series minimum driver remains 570+.

Sources: [NVIDIA CUDA 12.9 download archive](https://developer.nvidia.com/cuda-12-9-0-download-archive), [PyTorch 2.11 release blog](https://pytorch.org/blog/pytorch-2-11-release-blog/)

---

## Quick Reference: Updated Compatibility Matrix

| PyTorch | Supported CUDA | Notes |
|---------|---------------|-------|
| **2.11.x** | 12.6, 12.8, 12.9, **13.0** | Latest; CUDA 13 is the new default |
| 2.10.x | 12.6, 12.8, 12.9, 13.0 | Docs say "latest"; now superseded |
| 2.9.x | 12.6, 12.8, 13.0 | Still recommended for full FA2+sage+xformers compat |
| 2.7.x | 12.4, 12.6, 12.8 | Minimum for Blackwell |

| Library | Docs version | Latest | CUDA 13 Windows | Notes |
|---------|-------------|--------|-----------------|-------|
| flash-attn 2 | 2.8.3 | 2.8.3 | Broken (ABI) | Use FA3 for cu130 |
| flash-attn 3 | — | available | **Works** | Official wheels on PyTorch CDN |
| sageattention | 2.2.0 | 2.2.0 | **Now fixed** | cu130.torch2.11 wheels available |
| xformers | 0.0.33-0.0.34 | **0.0.35** | Works | Stable ABI; PyTorch 2.10+ compatible |
| triton-windows | 3.6.0 | 3.6.0 | Works | Repo moved to triton-lang org |

---

## Actionable Recommendations

1. **Update PyTorch version** in the recommended config from `2.9.1+cu128` to `2.10.0+cu128` (stable) or `2.11.0+cu130` (latest). Document that CUDA 13 is now the PyTorch default.

2. **Add Flash Attention 3** as the recommended path for CUDA 13 on Windows, referencing `download.pytorch.org/whl/flash-attn-3/` and the windreamer community wheels.

3. **Update SageAttention CUDA 13 status** from ❌ to ✅ — cu130 ABI3 wheels now work with PyTorch 2.10/2.11.

4. **Bump xFormers** from `0.0.34` (cu130) to `0.0.35`, and note the stable API/ABI change (one build, multiple PyTorch versions).

5. **Update Triton source link** from `woct0rdho/triton-windows` to `triton-lang/triton-windows` (the woct0rdho repo is archived). `pip install triton-windows` still works.

6. **Add CUDA 12.9** row to the toolkit table, and note that PyTorch 2.11 defaults to CUDA 13 when installing without an explicit index URL.

7. **Add PyTorch 2.11 FlexAttention / FA4 backend** note under the RTX 50-series section — Blackwell users on PyTorch 2.11 can get better performance via FlexAttention without needing a separate flash-attn 3 install.
