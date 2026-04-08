---
title: Customer Service Bot — OpenEnv
emoji: 🤖
colorFrom: gray
colorTo: blue
sdk: docker
app_port: 8000
---

# Customer Service Bot — OpenEnv

Run everything from the repository root:

```text
PS C:\NPersonal\Projects\openenv-project>
```

## Prerequisites

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or `pip`
- Optional: [Docker](https://docs.docker.com/get-docker/) for container builds
- Optional: [Ollama](https://ollama.com/) for local LLM (OpenAI-compatible API on port 11434)

## Install (root)

```powershell
cd C:\NPersonal\Projects\openenv-project
uv sync
```

Or: `python -m pip install -r requirements.txt`

If a leftover empty `RUNTIME\` folder exists from an older layout, close any program using it and delete the folder.

## Start the API server (`server/app.py`)

```powershell
uv run uvicorn server.app:app --host 127.0.0.1 --port 8000
```

Health check: `http://127.0.0.1:8000/health`

## Baseline inference (`inference.py`)

Requires the server above. From the same repo root:

```powershell
$env:ENV_BASE_URL = "http://127.0.0.1:8000"
$env:API_BASE_URL = "http://127.0.0.1:11434/v1"
$env:MODEL_NAME = "llama3.2"
$env:HF_TOKEN = "ollama"   # required for LLM path; no Python fallback — unset ⇒ rule-based only

uv run python inference.py
```

Structured logs go to **stdout** (`[START]`, `[STEP]`, `[END]`). Details go to **stderr**.

To print the **full chat transcript** while keeping graded stdout clean: `$env:VERBOSE_CHAT = "1"` before running `inference.py` (transcript on stderr only).

### Hugging Face / remote OpenAI-compatible

Set submission variables as required by the hackathon:

- `API_BASE_URL` — LLM base URL  
- `MODEL_NAME` — model id  
- `HF_TOKEN` — API key  

## Docker (build from root only)

```powershell
docker build -t openenv-runtime .
docker run --rm -p 8000:8000 openenv-runtime
```

The image uses **uv** to lock and install dependencies (`Dockerfile`).  
If inference runs **inside** Docker but Ollama runs on the host (Windows/macOS), use e.g. `API_BASE_URL=http://host.docker.internal:11434/v1` for the LLM client.

## Tests

```powershell
uv run python tests/test_environment.py
```

## Layout

| Path | Purpose |
|------|---------|
| `openenv.yaml` | OpenEnv metadata |
| `models.py` | Pydantic types |
| `server/app.py` | FastAPI app (`/reset`, `/step`, `/state`, `/health`) |
| `server/environment.py` | `CustomerServiceEnv` |
| `src/` | Tasks, graders, simulation |
| `inference.py` | Baseline inference (root; required for submission) |
| `Dockerfile` | Production image (uv + uvicorn) |

## Push to Hugging Face (`openenv push`)

From the repo root, pass **`.`** as the directory (last argument). Example:

```powershell
openenv push --repo-id YOUR_USER/YOUR_SPACE .
```

Log in first: `huggingface-cli login` (or set `HF_TOKEN`).

### README metadata (`colorFrom` / `colorTo`)

Hugging Face only accepts certain theme colors in the YAML above (e.g. `gray`, `red`, `yellow`, `green`, `blue`, `indigo`, `purple`, `pink`). If upload fails with **Invalid metadata in README.md**, fix those two fields—do not use custom color names or hex.

### Windows: `'charmap' codec can't encode character` (emoji in output)

PowerShell’s default encoding can break `openenv` when it prints Unicode (e.g. icons). Use UTF-8 for that session, then push:

```powershell
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
openenv push --repo-id YOUR_USER/YOUR_SPACE .
```

Or use the helper (same env vars):

```powershell
.\push-to-hf.ps1 push --repo-id YOUR_USER/YOUR_SPACE .
```

## Compliance

See `COMPLIANCE.md` for the pre-submission checklist (HF Space, Docker build, `openenv validate`, inference format, infra limits).
