# Migration : PyTorch 2.9.1+cu128 → 2.10.0+cu130

**Matériel cible** : NVIDIA RTX 5070 Ti (SM120, Blackwell), Windows 11, Python 3.12  
**Date de rédaction** : 2026-05-04  
**Basé sur** : [findings.md](findings.md) — recherches et vérifications de mai 2026

---

## Pourquoi 2.10.0+cu130 et pas 2.11.0 ?

- **FA4 / FlexAttention** de PyTorch 2.11 ne tourne **pas** sur SM120 — il exige SM100 (B200 datacenter). Aucun gain pour la 5070 Ti.
- Le wheel Flash Attention 2 pour `cu130 + torch2.11.0 + Python 3.12` n'est **pas confirmé** à ce jour (seul Python 3.13 l'est).
- PyTorch 2.10.0+cu130 est la combinaison où **tous** les composants ont un wheel Python 3.12 Windows vérifié.

Dès qu'un wheel FA2 cu130+torch2.11+cp312 apparaît chez Wildminder ou ussoewwin, la migration vers 2.11 sera triviale — seule la ligne `torch==` changera.

---

## Vue d'ensemble

| Composant | Avant | Après |
|-----------|-------|-------|
| CUDA Toolkit | 12.8 | **13.0** |
| PyTorch | 2.9.1+cu128 | **2.10.0+cu130** |
| torchvision | 0.24.1+cu128 | **0.25.0+cu130** |
| torchaudio | 2.9.1+cu128 | **2.10.0+cu130** |
| flash-attn | 2.8.3+cu128 | **2.8.3+cu130torch2.10.0** |
| sageattention | 2.2.0 (post3) | **2.2.0 post4 ABI3** (woct0rdho) |
| xformers | 0.0.33.post2 | **0.0.35** |
| triton-windows | 3.6.0 | 3.6.0 (inchangé) |
| venv | `.venv` | `.venv210` ← nouveau, `.venv` intact |

**Principe** : le venv existant (`.venv`) n'est jamais touché. Il reste le filet de sécurité jusqu'à validation complète.

---

## Phase 0 — Prérequis

### 0.1 Vérifier le driver NVIDIA

```powershell
nvidia-smi
```

Le driver doit être ≥ 570. La 5070 Ti avec driver 580.97 est ✅ — rien à faire.

### 0.2 Installer le CUDA Toolkit 13.0

Même si PyTorch cu130 embarque son propre runtime CUDA, les extensions (flash-attn, sageattention) ont besoin que les variables d'environnement pointent vers une installation cohérente pour éviter les conflits de DLL.

Télécharger : **https://developer.nvidia.com/cuda-13-0-0-download-archive**  
→ Windows → x86_64 → exe (local)

Après installation, vérifier dans un nouveau terminal :

```powershell
nvcc --version
# Doit afficher : release 13.0
```

### 0.3 Aligner les variables d'environnement

Dans **Panneau de configuration → Système → Variables d'environnement** (pour rendre permanent) :

| Variable | Valeur |
|----------|--------|
| `CUDA_PATH` | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0` |
| `PATH` (ajouter en tête) | `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin` |

Ou temporairement dans le terminal courant :

```powershell
$env:CUDA_PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0"
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\bin;" + $env:PATH
```

> ⚠️ Si CUDA 12.8 était dans le PATH, s'assurer que la v13.0 le précède.

---

## Phase 1 — Créer le nouveau venv

Depuis la racine du projet Scratchy :

```powershell
py -3.12 -m venv .venv210
.venv210\Scripts\activate
python --version
# Doit afficher : Python 3.12.x
```

---

## Phase 2 — PyTorch 2.10.0 + CUDA 13.0

```powershell
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
```

**Checkpoint obligatoire — ne pas continuer si ce test échoue :**

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"
```

Sortie attendue :
```
2.10.0+cu130
True
13.0
```

Si `cuda.is_available()` retourne `False` :
1. Vérifier que le driver est bien ≥ 570 (`nvidia-smi`)
2. Vérifier `CUDA_PATH` et `PATH`
3. Redémarrer le PC et réessayer
4. Ne pas passer à la phase suivante

---

## Phase 3 — Triton

Le dépôt source `woct0rdho/triton-windows` a été archivé le 2026-02-18. Le développement a migré vers `triton-lang/triton-windows`, mais le package PyPI (`triton-windows`) est identique — `pip install triton-windows` fonctionne sans changement.

```powershell
pip install triton-windows
```

```powershell
python -c "import triton; print(triton.__version__)"
# Attendu : 3.6.0
```

---

## Phase 4 — xFormers 0.0.35

Depuis la version 0.0.35, xFormers est compilé contre l'ABI stable de PyTorch 2.10+ : un seul wheel couvre PyTorch 2.10, 2.11, et les versions suivantes.

```powershell
pip install xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130
```

> ⚠️ Ne pas faire `pip install xformers` sans `--index-url` — pip pourrait tirer une version cu128 ou mettre à jour PyTorch de façon inattendue.

```powershell
python -c "import xformers; print(xformers.__version__)"
# Attendu : 0.0.35
```

---

## Phase 5 — SageAttention 2.2.0 post4 ABI3

> ⚠️ **Important** : le wheel Wildminder `sageattention-2.2.0+cu130torch2.10.0-cp312-cp312-win_amd64.whl` (post3) plante encore à l'import avec `DLL load failed while importing _fused`. Utiliser **exclusivement** le wheel post4 de woct0rdho ci-dessous.

Le wheel post4 utilise libtorch stable ABI + ABI3 et couvre PyTorch ≥ 2.9 — il n'y a pas besoin d'un build spécifique par version de torch.

```powershell
pip install "https://github.com/woct0rdho/SageAttention/releases/download/v2.2.0-windows.post4/sageattention-2.2.0+cu130torch2.9.0andhigher.post4-cp39-abi3-win_amd64.whl"
```

```powershell
python -c "import sageattention; print('sageattention OK')"
# Note : le package n'expose pas __version__, c'est normal
```

---

## Phase 6 — Flash Attention 2.8.3 (cu130 + torch2.10.0 + Python 3.12)

### 6.1 Trouver le wheel exact sur Wildminder

Le nom du fichier peut inclure un date-stamp (`d20260121`). Avant de lancer pip, **vérifier le nom exact** :

👉 **https://huggingface.co/Wildminder/AI-windows-whl/tree/main**

Rechercher dans la liste : `cu130torch2.10.0` + `cp312`  
Le fichier devrait ressembler à :
```
flash_attn-2.8.3+cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
```
ou avec un date-stamp :
```
flash_attn-2.8.3+d20260121.cu130torch2.10.0cxx11abiTRUE-cp312-cp312-win_amd64.whl
```

### 6.2 Installation

Copier le lien du fichier depuis HuggingFace et l'utiliser directement :

```powershell
pip install "<URL copiée depuis Wildminder>"
```

```powershell
python -c "import flash_attn; print(flash_attn.__version__)"
# Attendu : 2.8.3
```

### 6.3 Fallback : Flash Attention 3

Si le wheel Wildminder n'est plus disponible ou si l'import échoue, Flash Attention 3 est une alternative valable pour cu130 sur SM120 (SM120 ≥ SM90 requis par FA3) :

```powershell
pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/
```

---

## Phase 7 — Scratchy

```powershell
pip install -e .
```

---

## Phase 8 — Vérification complète

```powershell
python checklibs.py
```

Sortie attendue :

```
python version: 3.12.x
torch version: 2.10.0+cu130
cuda version (torch): 13.0
torchvision version: 0.25.0+cu130
torchaudio version: 2.10.0+cu130
cuda available: True
flash-attention version: 2.8.3
triton version: 3.6.0
sageattention is installed but has no __version__ attribute
xformers: 0.0.35
```

Lancer les tests unitaires :

```powershell
pytest tests/unit -v
```

---

## Phase 9 — Basculer

Une fois les tests validés, le nouveau venv est opérationnel. Mettre à jour les instructions d'activation dans ton workflow habituel (scripts, README local, etc.) pour pointer sur `.venv210`.

Garder `.venv` (cu128) quelques semaines comme filet de sécurité avant de le supprimer :

```powershell
# Quand tu es prêt à supprimer l'ancien venv :
Remove-Item -Recurse -Force .venv
```

---

## Rollback

À n'importe quelle phase, retour à l'état initial en une commande :

```powershell
deactivate
.venv\Scripts\activate
```

Si la migration a mal tourné et que tu veux repartir de zéro :

```powershell
deactivate
Remove-Item -Recurse -Force .venv210
# Reprendre depuis la Phase 1
```

---

## Checklist rapide

- [ ] Driver ≥ 570 confirmé (`nvidia-smi`)
- [ ] CUDA Toolkit 13.0 installé (`nvcc --version`)
- [ ] `CUDA_PATH` et `PATH` mis à jour vers v13.0
- [ ] `.venv210` créé
- [ ] `torch.cuda.is_available()` → `True` avec cuda 13.0
- [ ] triton 3.6.0 importé
- [ ] xformers 0.0.35 importé
- [ ] sageattention post4 importé sans erreur
- [ ] flash-attn 2.8.3 importé sans erreur
- [ ] `pip install -e .` réussi
- [ ] `checklibs.py` → toutes les lignes correctes
- [ ] `pytest tests/unit` → vert
- [ ] `.venv` conservé jusqu'à validation en production

---

## Ressources

- [Wildminder Windows wheels](https://huggingface.co/Wildminder/AI-windows-whl/tree/main)
- [woct0rdho/SageAttention releases](https://github.com/woct0rdho/SageAttention/releases)
- [windreamer Flash Attention 3 wheels](https://windreamer.github.io/flash-attention3-wheels/)
- [triton-lang/triton-windows](https://github.com/triton-lang/triton-windows)
- [CUDA 13.0 download](https://developer.nvidia.com/cuda-13-0-0-download-archive)
- [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/)
- [CUDA_COMPATIBILITY.md](CUDA_COMPATIBILITY.md) — doc de référence du projet
- [findings.md](findings.md) — notes de recherche détaillées (mai 2026)
