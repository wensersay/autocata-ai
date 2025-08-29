FROM python:3.10-slim

# Dependencias nativas
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-spa poppler-utils \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
ARG CACHEBUST=2025-09-01a
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# No fijamos PORT aquí. Railway define $PORT en runtime.
EXPOSE 8080

# Importante: atarse al $PORT real que inyecta Railway
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
