# Scratchy

Production-ready AI Image Generation API Server with authentication, credit-based billing, and job management.

## Features

- **Multiple Models**: FLUX.1-schnell, FLUX.1-dev, Z-Image-Turbo, SDXL
- **API Key Authentication**: Secure access with hashed keys
- **Credit System**: Pay-per-image billing with refunds on failure
- **Rate Limiting**: Per-key configurable limits
- **Job Queue**: Bounded FIFO queue with SSE progress streaming
- **Result Storage**: Temporary storage with configurable TTL
- **Webhook Notifications**: Async job completion callbacks
- **Admin API**: Key management, analytics, backups
- **RFC 7807 Errors**: Standardized error responses

## Quick Start

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For Z-Image-Turbo (requires diffusers from source)
pip install git+https://github.com/huggingface/diffusers
```

### 2. Configure

```bash
cp config.example.yaml config.yaml
# Edit config.yaml as needed
```

### 3. Create an API Key

```bash
python -m scratchy.cli.keys create --name "my_app" --credits 100
```

**Save the displayed API key - it won't be shown again!**

### 4. Start the Server

```bash
python -m scratchy.main
```

Server starts at `http://localhost:8080`

The first launch downloads the model (~10-20 GB depending on model).

## API Documentation

Interactive docs available at:
- **Swagger UI**: http://localhost:8080/docs
- **ReDoc**: http://localhost:8080/redoc

### Authentication

All endpoints (except health checks) require an API key:

```bash
curl -H "Authorization: Bearer sk_your_key_here" http://localhost:8080/v1/account/balance
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/health/live` | GET | Liveness probe |
| `/v1/health/ready` | GET | Readiness probe (model, DB, queue) |
| `/v1/generate` | POST | Generate image (JSON response) |
| `/v1/generate/raw` | POST | Generate image (binary response) |
| `/v1/generate/stream` | POST | Generate with SSE progress |
| `/v1/jobs/{id}` | GET | Retrieve job result |
| `/v1/jobs/{id}` | DELETE | Cancel job |
| `/v1/account/balance` | GET | Check credit balance |
| `/v1/admin/keys` | GET/POST | List/create API keys |
| `/v1/admin/keys/{id}` | GET/PUT/DELETE | Manage API key |
| `/v1/admin/analytics` | GET | Usage analytics |
| `/v1/admin/backup` | POST | Trigger backup |

### Generate Image

```bash
curl -X POST http://localhost:8080/v1/generate \
  -H "Authorization: Bearer sk_your_key" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A serene mountain landscape at sunset",
    "width": 1024,
    "height": 1024,
    "steps": 4,
    "output_format": "png"
  }'
```

Response:
```json
{
  "job_id": "job_20240115_12345",
  "status": "completed",
  "image": "base64...",
  "seed": 42,
  "generation_time": 2.34,
  "warnings": [],
  "credits_used": 1,
  "credits_remaining": 99
}
```

### Generate with Progress (SSE)

```bash
curl -N http://localhost:8080/v1/generate/stream \
  -H "Authorization: Bearer sk_your_key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A cat in space"}'
```

Events:
```
event: queued
data: {"job_id": "job_abc123", "position": 1}

event: started
data: {"job_id": "job_abc123"}

event: progress
data: {"step": 2, "total_steps": 4}

event: completed
data: {"job_id": "job_abc123", "retrieval_url": "/v1/jobs/job_abc123"}
```

### Raw Binary Response

```bash
curl -X POST http://localhost:8080/v1/generate/raw \
  -H "Authorization: Bearer sk_your_key" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "A sunset"}' \
  --output image.png
```

Metadata returned in headers:
- `X-Scratchy-Seed`
- `X-Scratchy-Generation-Time`
- `X-Scratchy-Credits-Remaining`

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | required | Image description |
| `negative_prompt` | string | null | What to avoid (SDXL) |
| `width` | int | 1024 | Width (256-2048, rounded to 64) |
| `height` | int | 1024 | Height (256-2048, rounded to 64) |
| `steps` | int | auto | Inference steps |
| `guidance_scale` | float | auto | CFG scale |
| `seed` | int | random | Seed for reproducibility |
| `output_format` | string | "png" | png, jpeg, webp |
| `webhook_url` | string | null | URL for completion callback |

## CLI Commands

### Key Management

```bash
# Create a key
python -m scratchy.cli.keys create --name "my_app" --credits 100 --rate-limit 20

# List keys
python -m scratchy.cli.keys list

# Show key details
python -m scratchy.cli.keys show <key_id>

# Add credits
python -m scratchy.cli.keys add-credits <key_id> 50 --reason "Monthly refill"

# Update key
python -m scratchy.cli.keys update <key_id> --rate-limit 30

# Deactivate key
python -m scratchy.cli.keys delete <key_id>
```

### Backup Management

```bash
# Create backup
python -m scratchy.cli.backup create

# List backups
python -m scratchy.cli.backup list

# Restore from backup
python -m scratchy.cli.backup restore scratchy_20240115_120000.db

# Cleanup old backups
python -m scratchy.cli.backup cleanup --days 7
```

## Configuration

Configuration is loaded in order (later overrides earlier):
1. Code defaults
2. `config.yaml` file
3. Environment variables (prefix: `SCRATCHY_`)

### config.yaml Example

```yaml
server:
  host: "0.0.0.0"
  port: 8080

model:
  name: flux-schnell  # flux-schnell, flux-dev, z-turbo, sdxl
  quantization: none  # none, 8bit, 4bit
  device: cuda        # cuda, mps, cpu

queue:
  max_depth: 10

storage:
  jobs_dir: "./scratchy_data/jobs"
  jobs_ttl_hours: 1
  db_path: "./scratchy_data/scratchy.db"

auth:
  default_rate_limit: 10  # requests per minute

security:
  cors_origins: ["*"]
  max_prompt_length: 2000
```

### Environment Variables

```bash
SCRATCHY_SERVER__HOST=0.0.0.0
SCRATCHY_SERVER__PORT=8080
SCRATCHY_MODEL__NAME=flux-schnell
SCRATCHY_MODEL__QUANTIZATION=8bit
SCRATCHY_MODEL__DEVICE=cuda
SCRATCHY_QUEUE__MAX_DEPTH=10
```

### Supported Models

| Model | Config Name | Steps | VRAM | License |
|-------|-------------|-------|------|---------|
| FLUX.1-schnell | `flux-schnell` | 4 | ~12 GB | Apache 2.0 |
| FLUX.1-dev | `flux-dev` | 28 | ~16 GB | Non-commercial |
| Z-Image-Turbo | `z-turbo` | 8 | <16 GB | Apache 2.0 |
| SDXL | `sdxl` | 30 | ~8 GB | CreativeML |

### Quantization

Reduce VRAM usage:

```yaml
model:
  name: flux-schnell
  quantization: 8bit  # Reduces VRAM by ~50%
```

## Credit System

- **Pricing**: 1 credit = 1 image (flat rate)
- **Deduction**: Credits deducted when generation starts
- **Refunds**: Automatic refund on generation failure
- **Cancellation**: Full refund when cancelling queued jobs

## Error Handling

All errors follow RFC 7807 Problem Details:

```json
{
  "type": "https://scratchy.api/errors/insufficient-credits",
  "title": "Insufficient Credits",
  "status": 402,
  "detail": "Your account has 0 credits. This request requires 1 credit.",
  "instance": "/v1/generate"
}
```

## Docker

```bash
# Build
docker build -t scratchy .

# Run
docker run -p 8080:8080 \
  -v ./config.yaml:/app/config.yaml:ro \
  -v scratchy-data:/var/scratchy \
  --gpus all \
  scratchy
```

Or with Docker Compose:

```bash
docker-compose up -d
```

## iOS/Swift Integration

```swift
import Foundation

struct ScratchyClient {
    let baseURL: String
    let apiKey: String

    func generate(prompt: String) async throws -> Data {
        var request = URLRequest(url: URL(string: "\(baseURL)/v1/generate/raw")!)
        request.httpMethod = "POST"
        request.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization")
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(["prompt": prompt])

        let (data, _) = try await URLSession.shared.data(for: request)
        return data
    }
}

// Usage
let client = ScratchyClient(baseURL: "http://YOUR_SERVER:8080", apiKey: "sk_xxx")
let imageData = try await client.generate(prompt: "A cute robot")
let image = UIImage(data: imageData)
```

## Python Client

```python
import httpx

client = httpx.Client(
    base_url="http://localhost:8080",
    headers={"Authorization": "Bearer sk_your_key"}
)

# Generate image
response = client.post("/v1/generate", json={
    "prompt": "A beautiful sunset",
    "width": 1024,
    "height": 1024,
})

result = response.json()
image_base64 = result["image"]
```

## Testing

```bash
# Unit tests
pytest tests/unit -v

# Integration tests (requires GPU)
pytest tests/integration -v

# Load tests
locust -f tests/load/locustfile.py --host=http://localhost:8080
```

## Legacy Server

The original simple server without authentication is still available at `server.py`.

```bash
python server.py
```

## License

MIT License
