# OpenEnv Demo Environment

This repository contains a demo OpenEnv environment with **separate runtime and training setups**.

---

## Project Structure

* `RUNTIME/demo/` → Environment + server
* `TRAINING/` → Training pipeline (separate environment)

---

## Environment Development

Edit core files inside:

* `RUNTIME/demo/server/demo_environment.py` → environment logic
* `RUNTIME/demo/models.py` → model definitions

---

## Running the Server (Canonical Flow)

```powershell
cd RUNTIME
cd demo
.\.venv\Scripts\activate

# docker build --no-cache -t openenv-runtime .
docker run -p 8000:8000 openenv-runtime
```

Access:

```
http://localhost:8000/web/
```

---

## Local Run (Preferred for Development)

Run from `RUNTIME` (not inside `demo`):

```powershell
cd RUNTIME
.\.venv\Scripts\activate
uvicorn demo.server.app:app --host 0.0.0.0 --port 8000
```

---

## Training (Canonical Flow)

```powershell
cd TRAINING
.\.venv\Scripts\activate
python .\train.py
```

---

## Notes

* `RUNTIME` and `TRAINING` are isolated environments
* Docker is optional; local `uvicorn` run is faster for iteration
* Server must be running on port `8000` before training