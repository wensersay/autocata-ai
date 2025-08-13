FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libc6 libstdc++6 libjpeg62-turbo \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./

ENV PYTHONUNBUFFERED=1
ENV PORT=8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
