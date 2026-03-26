FROM python:3.10

WORKDIR /app

# Copy everything
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 7860

# Run app
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]