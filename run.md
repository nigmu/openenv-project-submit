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