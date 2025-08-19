FROM python:3.10-slim

# Dependencias nativas (tesseract/poppler y libs de OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
# Bumpea CACHEBUST para forzar rebuild cuando lo necesites
ARG CACHEBUST=2025-08-19c
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Railway suele inyectar PORT=8080; nos adaptamos a lo que venga
ENV PORT=8080

EXPOSE 8080

# Importante: usa el $PORT que da Railway (o 8000 si no existe)
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
