# 🎨 Image Generation API Server

Serveur API minimaliste pour la génération d'images avec diffusers.
Compatible avec FLUX.1, Z-Image-Turbo, SDXL et autres modèles.

## 📋 Prérequis

- Python 3.10+
- CUDA toolkit (pour GPU NVIDIA)
- ~16 Go de VRAM pour la plupart des modèles

## 🚀 Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou: venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Pour Z-Image-Turbo (nécessite diffusers depuis source)
pip install git+https://github.com/huggingface/diffusers
```

## ⚙️ Configuration

Édite `server.py` et modifie `MODEL_CONFIG` pour choisir ton modèle :

```python
MODEL_CONFIG = {
    # FLUX.1-schnell - Ultra rapide, Apache 2.0
    "name": "black-forest-labs/FLUX.1-schnell",
    "pipeline": "flux",
    "default_steps": 4,
    "guidance_scale": 0.0,
}
```

### Modèles disponibles

| Modèle | Steps | VRAM | Licence | Notes |
|--------|-------|------|---------|-------|
| FLUX.1-schnell | 4 | ~12 Go | Apache 2.0 | Ultra rapide |
| FLUX.1-dev | 28 | ~16 Go | Non-commercial | Meilleure qualité |
| Z-Image-Turbo | 8 | <16 Go | Apache 2.0 | Excellent rendu texte |
| SDXL | 30 | ~8 Go | CreativeML | Écosystème LoRA |

## 🏃 Lancement

```bash
python server.py
```

Le serveur démarre sur `http://localhost:8080`.

Le premier lancement télécharge le modèle (~10-20 Go selon le modèle).

## 📡 API

### Health Check

```bash
curl http://localhost:8080/health
```

### Génération (réponse JSON avec base64)

```bash
curl -X POST http://localhost:8080/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cute robot painting a sunset, digital art",
    "width": 1024,
    "height": 1024,
    "steps": 8
  }'
```

### Génération (réponse binaire directe)

```bash
curl -X POST http://localhost:8080/generate/raw \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a cat in space"}' \
  --output image.png
```

### Paramètres

| Paramètre | Type | Défaut | Description |
|-----------|------|--------|-------------|
| `prompt` | string | requis | Description de l'image |
| `negative_prompt` | string | null | Ce qu'on ne veut pas (SDXL) |
| `width` | int | 1024 | Largeur (256-2048) |
| `height` | int | 1024 | Hauteur (256-2048) |
| `steps` | int | auto | Nombre de steps |
| `guidance_scale` | float | auto | CFG scale |
| `seed` | int | random | Seed pour reproductibilité |
| `output_format` | string | "png" | png, jpeg, webp |

## 📱 Exemple Swift (iOS)

```swift
import Foundation

struct GenerateRequest: Codable {
    let prompt: String
    let width: Int
    let height: Int
    let steps: Int?
}

struct GenerateResponse: Codable {
    let image_base64: String
    let seed: Int
    let generation_time_ms: Int
}

func generateImage(prompt: String) async throws -> UIImage {
    let url = URL(string: "http://YOUR_SERVER:8080/generate")!
    var request = URLRequest(url: url)
    request.httpMethod = "POST"
    request.setValue("application/json", forHTTPHeaderField: "Content-Type")
    
    let body = GenerateRequest(prompt: prompt, width: 1024, height: 1024, steps: nil)
    request.httpBody = try JSONEncoder().encode(body)
    
    let (data, _) = try await URLSession.shared.data(for: request)
    let response = try JSONDecoder().decode(GenerateResponse.self, from: data)
    
    guard let imageData = Data(base64Encoded: response.image_base64),
          let image = UIImage(data: imageData) else {
        throw NSError(domain: "ImageError", code: -1)
    }
    
    return image
}
```

## 🔧 Optimisations

### Réduire l'utilisation VRAM

```python
# Dans server.py, après le chargement du pipeline:
pipe.enable_attention_slicing()
pipe.enable_vae_slicing()
pipe.enable_model_cpu_offload()  # Décharge sur CPU si besoin
```

### Accélérer avec xformers

```bash
pip install xformers
```

```python
pipe.enable_xformers_memory_efficient_attention()
```

### Quantification (modèles GGUF via ComfyUI)

Pour encore moins de VRAM, utilise les versions quantifiées via ComfyUI en mode API.

## 🐳 Docker (optionnel)

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

# Pre-download model (optionnel, sinon téléchargé au premier run)
# RUN python -c "from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell')"

EXPOSE 8080
CMD ["python", "server.py"]
```

```bash
docker build -t image-api .
docker run --gpus all -p 8080:8080 image-api
```

## 📝 Notes

- Le premier appel est lent (chargement du modèle en VRAM)
- Les appels suivants sont rapides (~1-5s selon modèle/steps)
- Pour une app iPhone, expose le serveur via un tunnel (ngrok, cloudflare) ou déploie sur un VPS avec GPU
