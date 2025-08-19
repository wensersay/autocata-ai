FROM python:3.10-slim

# Dependencias del sistema (Tesseract + Poppler + libs de OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Evitar .pyc y usar stdout sin buffer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Truco para forzar rebuild de capas cuando quieras
ARG CACHEBUST=2025-08-19b

# Instalar Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código
COPY . .

# Exponer 8000 para ejecución local (Railway ignora EXPOSE, usa $PORT)
EXPOSE 8000

# NO fijes el puerto a 8000: usa el PORT que inyecta Railway; 8000 es fallback local
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
