FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1
# Render fournit le port dans la variable PORT
CMD uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-10000}
