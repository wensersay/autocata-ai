# AutoCatastro AI (microservicio base)

## Ejecutar en local
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app:app --reload
```
Abre `http://127.0.0.1:8000/health` → `{"ok": true}`.

### Probar `/extract`
```bash
curl -X POST http://127.0.0.1:8000/extract \
  -H "Content-Type: application/json" \
  -d '{"pdf_url":"https://autocatastro.com/wp-content/uploads/2025/08/archivo.pdf"}'
```

## Docker (local)
```bash
docker build -t autocata-ai:0.1 .
docker run -p 8000:8000 autocata-ai:0.1
```

## Desplegar (Railway/Render/Fly/Cloud Run)
1. Crea un repo en GitHub con estos 4 archivos.
2. **Railway**: New Project → Deploy from Repo → detecta Dockerfile → Deploy.
3. **Render**: New → Web Service → Repo → Docker → Port 8000.
4. **Cloud Run**: `gcloud run deploy` apuntando a la imagen (previo push a Artifact/Container Registry).

URL del endpoint: `https://TU-APP.railway.app/extract`

## Integración con WordPress
En `autocatastro-plugin.php` define:
```php
if (!defined('AC_AI_ENDPOINT')) {
    define('AC_AI_ENDPOINT', 'https://TU-APP.railway.app/extract');
}
```
Tu `Processor.php` ya llama a `AC_AI_ENDPOINT` y completará linderos si el servicio responde.
