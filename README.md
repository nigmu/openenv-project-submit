GRPO training and remote-LLM evaluation live in `trainer/` (separate `uv` project; not in the server Docker image).

Next steps:
  cd C:\NPersonal\Projects\openenv-project\demo
  # Edit your environment implementation in server/demo_environment.py
  # Edit your models in models.py
  # Install dependencies: uv sync

  # To integrate into OpenEnv repo:
  # 1. Copy this directory to <repo_root>/envs/demo_env
  # 2. Build from repo root: docker build -t demo_env:latest -f envs/demo_env/server/Dockerfile .
  # 3. Run your image: docker run -p 8000:8000 demo_env:latest

  1. c:\NPersonal\Projects\openenv-project\venv\Scripts\Activate.ps1
2. cd .\demo\ 
<!-- make docker image: docker build -t demo:latest -f Dockerfile .  -->
3. docker run -p 8000:8000 demo:latest 
4. http://localhost:8000/web/

LOCAL RUN
1. c:\NPersonal\Projects\openenv-project\venv\Scripts\Activate.ps1  
2. cd .\demo\ 
3. set PYTHONPATH=%cd%
4. uvicorn server.app:app --host 0.0.0.0 --port 8000
<!-- in separate terminal -->
5. python .\test_client.py

TRAINING / REMOTE LLM EVAL (separate project, own venv — not in Docker image)
1. cd ..\trainer\
2. uv sync
3. Start server as above on port 8000
4. uv run train-math-env   OR   uv run infer-math-env



cd RUNTIME
docker build --no-cache -t openenv-runtime .
docker run -p 8000:8000 openenv-runtime
