Next steps:
  cd C:\NPersonal\Projects\openenv-project\demo
  # Edit your environment implementation in server/demo_environment.py
  # Edit your models in models.py
  # Install dependencies: uv sync

  # To integrate into OpenEnv repo:
  # 1. Copy this directory to <repo_root>/envs/demo_env
  # 2. Build from repo root: docker build -t demo_env:latest -f envs/demo_env/server/Dockerfile .
  # 3. Run your image: docker run -p 8000:8000 demo_env:latest