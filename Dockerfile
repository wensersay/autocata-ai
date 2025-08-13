FROM python:3.11-slim

# Dependencias del sistema para pdf2image (poppler), Tesseract y OpenCV headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils tesseract-ocr tesseract-ocr-spa \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./ 
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Escucha el puerto que asigne Railway; fallback 8000
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
