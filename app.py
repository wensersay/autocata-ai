#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
autoCata microservice (app.py)
- FastAPI with JWT Bearer auth (Swagger "Authorize" button)
- /extract: OCR (Tesseract) + simple visual/text heuristics for linderos & titulares
- strict + fallback pipeline (strict default) with env-tunable thresholds
- GPT-4o notarial drafting via OpenAI API (single call, after result chosen)
- /download/{filename}: public, returns .txt notarial text
- Returns JSON with notarial_text + download_url, as requested
"""

import os
import io
import re
import json
import time
import math
import base64
import logging
from typing import Dict, List, Optional, Tuple, Any

from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, PlainTextResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.requests import Request
from starlette.status import HTTP_401_UNAUTHORIZED
from starlette.staticfiles import StaticFiles

import jwt

# OCR / image
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image, ImageOps, ImageFilter
import numpy as np
import cv2

# OpenAI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ----------------------
# Logging
# ----------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("autoCata")

# ----------------------
# Environment & defaults
# ----------------------

def _bool_env(name: str, default: bool = False) -> bool:
    v = os.getenv(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default

def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

# Security / JWT
JWT_SECRET = os.getenv("JWT_SECRET", "CHANGE_ME_SECRET")
JWT_ALG = os.getenv("JWT_ALG", "HS256")
JWT_AUDIENCE = os.getenv("JWT_AUD", None) or None
JWT_ISSUER = os.getenv("JWT_ISS", None) or None

# Strict+fallback defaults
STRICT_MODE_DEFAULT = _bool_env("STRICT_MODE_DEFAULT", True)
FALLBACK_ON_STRICT_FAIL = _bool_env("FALLBACK_ON_STRICT_FAIL", True)
STRICT_MIN_CONF = _float_env("STRICT_MIN_CONF", 0.82)
STRICT_REQUIRE_4_SIDES = _bool_env("STRICT_REQUIRE_4_SIDES", True)

# OCR / DPI
PDF_DPI = _int_env("PDF_DPI", 300)
SLOW_DPI = _int_env("SLOW_DPI", 400)
FAST_DPI = _int_env("FAST_DPI", 200)
AUTO_DPI = _bool_env("AUTO_DPI", True)

# Misc heuristics
NAME_HINTS = _bool_env("NAME_HINTS", True)
NAME_HINTS_FILE = os.getenv("NAME_HINTS_FILE", "")
NAME_HINTS_EXTRA = os.getenv("NAME_HINTS_EXTRA", "")
NEIGH_MIN_AREA_HARD = _float_env("NEIGH_MIN_AREA_HARD", 0.01)  # fraction of page area

# Output dir for txt
DOWNLOAD_DIR = os.getenv("DOWNLOAD_DIR", "downloads")
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# GPT model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Reject un-OCRable PDFs?
REJECT_NO_OCR = _bool_env("REJECT_NO_OCR", True)
MIN_OCR_CHARS = _int_env("MIN_OCR_CHARS", 200)

# Debug mode
DIAG_MODE = _bool_env("DIAG_MODE", True)

# ----------------------
# FastAPI app + Swagger security
# ----------------------

app = FastAPI(
    title="autoCata Microservice",
    version="0.7.0",
    description="OCR + extracción de colindantes y redacción notarial",
)

# CORS (adjust for your domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bearer_scheme = HTTPBearer(auto_error=False)

def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> dict:
    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Authorization header missing or invalid")

    token = credentials.credentials
    try:
        options = {"verify_aud": JWT_AUDIENCE is not None}
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALG],
            audience=JWT_AUDIENCE if JWT_AUDIENCE else None,
            issuer=JWT_ISSUER if JWT_ISSUER else None,
            options=options,
        )
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")

# Expose Bearer in OpenAPI so Swagger shows "Authorize"
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    from fastapi.openapi.utils import get_openapi
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    openapi_schema.setdefault("components", {}).setdefault("securitySchemes", {})
    openapi_schema["components"]["securitySchemes"]["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
    }
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# ----------------------
# Utilities
# ----------------------

def load_name_hints() -> List[str]:
    hints = []
    if NAME_HINTS_FILE and os.path.exists(NAME_HINTS_FILE):
        try:
            with open(NAME_HINTS_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    t = line.strip()
                    if t:
                        hints.append(t)
        except Exception as e:
            log.warning(f"NAME_HINTS_FILE read error: {e}")
    if NAME_HINTS_EXTRA:
        for h in NAME_HINTS_EXTRA.split(","):
            h = h.strip()
            if h:
                hints.append(h)
    return hints

NAME_HINTS_LIST = load_name_hints()

def choose_dpi(pdf_bytes: bytes) -> int:
    if not AUTO_DPI:
        return PDF_DPI
    # crude heuristic using file size
    n = len(pdf_bytes)
    if n < 500_000:
        return SLOW_DPI  # small pdf -> maybe low res, upscale a bit
    if n > 5_000_000:
        return FAST_DPI  # big pdf -> try faster
    return PDF_DPI

def pdf_to_images(pdf_bytes: bytes, dpi: int) -> List[Image.Image]:
    try:
        imgs = convert_from_bytes(pdf_bytes, dpi=dpi, fmt="png", thread_count=2)
        return imgs
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"No se pudo rasterizar el PDF: {e}")

def preprocess_image(img: Image.Image) -> Image.Image:
    # grayscale, slight contrast/binarization
    g = ImageOps.grayscale(img)
    # unsharp and slight thresholding can help OCR
    g = g.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
    return g

def ocr_image(img: Image.Image) -> str:
    try:
        cfg = "--oem 3 --psm 6 -l spa+eng"
        txt = pytesseract.image_to_string(img, config=cfg)
        return txt or ""
    except Exception as e:
        log.warning(f"OCR error: {e}")
        return ""

def quick_ocr(pdf_bytes: bytes) -> Tuple[str, List[Image.Image]]:
    dpi = choose_dpi(pdf_bytes)
    imgs = pdf_to_images(pdf_bytes, dpi=dpi)
    all_text = []
    for i, im in enumerate(imgs[:4]):  # first pages usually enough
        pim = preprocess_image(im)
        t = ocr_image(pim)
        if t:
            all_text.append(t)
    return "\n".join(all_text).strip(), imgs

# Very simple visual neighbor detection placeholder (contour areas around target polygon color heuristics)
def detect_neighbors_visual(img: Image.Image, strict: bool) -> Dict[str, Any]:
    # Placeholder: in real pipeline, detect target parcel (green/teal) and neighbors (pink) by color masks.
    # Here we just return empty and let text heuristics drive.
    return {"neighbors": {}, "confidence": 0.0, "debug": {"visual_used": False}}

# Text heuristics for linderos
CARDINAL_KEYS = {
    "norte": ["norte", "n."],
    "sur": ["sur", "s."],
    "este": ["este", "e."],
    "oeste": ["oeste", "o."],
    # extended
    "noreste": ["noreste", "ne"],
    "sureste": ["sureste", "se"],
    "suroeste": ["suroeste", "so"],
    "noroeste": ["noroeste", "no"],
}

SIDE_PATTERN = re.compile(r"(norte|sur|este|oeste|noreste|sureste|suroeste|noroeste)\s*[:\-–]\s*(.+?)(?=$|\n|\.|\;)", re.I)

def extract_sides_from_text(text: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    # common Spanish phrasing "Linda al norte con X, al sur con Y..."
    # First, structured lines "NORTE: X"
    for m in SIDE_PATTERN.finditer(text):
        side = m.group(1).lower()
        val = m.group(2).strip()
        # trim trailing punctuation
        val = re.sub(r"[\.;,\s]+$", "", val)
        out[side] = val

    # Then, phrasing "Linda al norte con XXX"
    linda_pattern = re.compile(r"(?:linda|lindando)\s+al\s+(norte|sur|este|oeste)\s+con\s+(.+?)(?=,|\.\s|$)", re.I)
    for m in linda_pattern.finditer(text):
        side = m.group(1).lower()
        val = m.group(2).strip()
        val = re.sub(r"[\.;,\s]+$", "", val)
        out.setdefault(side, val)

    return out

PROPER_NAME_WORD = re.compile(r"\b([A-ZÁÉÍÓÚÑ]{2,}(?:\s+[A-ZÁÉÍÓÚÑ]{2,}){1,4})\b")

def detect_owners(text: str) -> List[str]:
    owners: List[str] = []
    # Hint-based
    if NAME_HINTS and NAME_HINTS_LIST:
        for h in NAME_HINTS_LIST:
            if h and re.search(re.escape(h), text, re.I):
                owners.append(h)
    # crude proper name detection (uppercase sequences)
    for m in PROPER_NAME_WORD.finditer(text):
        cand = m.group(1).strip()
        if len(cand.split()) <= 6 and len(cand) >= 6:
            owners.append(cand)
    # unique preserve order
    seen = set()
    uniq = []
    for o in owners:
        if o not in seen:
            uniq.append(o)
            seen.add(o)
    return uniq[:20]

def fuse_results(text: str, visual: Dict[str, Any], strict: bool) -> Dict[str, Any]:
    sides = extract_sides_from_text(text)
    owners = detect_owners(text)
    # Confidence estimate very naive
    # +0.25 per principal side, capped at 1.0
    score = 0.0
    for k in ["norte", "sur", "este", "oeste"]:
        if sides.get(k):
            score += 0.25
    score = min(1.0, score)

    return {
        "linderos": sides,
        "owners_detected": owners,
        "confidence": score,
        "note": None,
        "debug": {
            "strict": strict,
            "len_text": len(text),
            "visual": visual.get("debug") if visual else None,
        }
    }

def is_valid_extraction(res: dict, strict: bool) -> bool:
    if not res:
        return False
    lind = (res or {}).get("linderos") or {}
    owners = (res or {}).get("owners_detected") or []
    conf = float((res or {}).get("confidence", 0.0))

    if strict:
        if conf < STRICT_MIN_CONF:
            return False
        if STRICT_REQUIRE_4_SIDES:
            needed = {"norte","sur","este","oeste"}
            if any(not lind.get(k) for k in needed):
                return False
        if len(owners) == 0:
            return False
        return True
    else:
        sides_present = [v for v in lind.values() if v]
        if len(sides_present) >= 2 or len(owners) >= 1 or conf >= 0.60:
            return True
        return False

def run_extraction_pipeline(pdf_bytes: bytes, strict: bool) -> Dict[str, Any]:
    text, imgs = quick_ocr(pdf_bytes)
    if REJECT_NO_OCR and len(text) < MIN_OCR_CHARS:
        # En estricto devolvemos vacío; el endpoint más arriba rechazará con 400 si no hay OCR suficiente
        return {
            "linderos": {},
            "owners_detected": [],
            "confidence": 0.0,
            "note": "OCR insuficiente",
            "debug": {"len_text": len(text), "strict": strict},
        }
    # Visual placeholder (could be extended to color-based neighbor detection)
    visual = {}
    if imgs:
        try:
            visual = detect_neighbors_visual(imgs[0], strict=strict)
        except Exception as e:
            if DIAG_MODE:
                log.warning(f"visual detection error: {e}")
            visual = {"neighbors": {}, "confidence": 0.0, "debug": {"visual_used": False, "error": str(e)}}

    fused = fuse_results(text, visual, strict=strict)
    return fused

def extract_with_strict_fallback(pdf_bytes: bytes) -> Tuple[Dict[str, Any], bool]:
    # Strict first
    res_strict = run_extraction_pipeline(pdf_bytes, strict=True)
    if is_valid_extraction(res_strict, strict=True):
        return res_strict, False
    # Fallback
    if FALLBACK_ON_STRICT_FAIL:
        res_fb = run_extraction_pipeline(pdf_bytes, strict=False)
        if is_valid_extraction(res_fb, strict=False):
            res_fb["note"] = ((res_fb.get("note") or "") + " [fallback]").strip()
            return res_fb, True
    # return best
    best = res_strict
    return best or {}, True

def draft_notarial_text(linderos: Dict[str, str]) -> str:
    # Sencilla frase notarial canónica si no se usa GPT
    norte = linderos.get("norte", "___")
    sur = linderos.get("sur", "___")
    este = linderos.get("este", "___")
    oeste = linderos.get("oeste", "___")
    return f"Linda al norte con {norte}; al sur con {sur}; al este con {este}; y al oeste con {oeste}."

async def generate_notarial_text_with_gpt(result: Dict[str, Any]) -> str:
    # Only GPT-4o if API key is present; else fallback to simple draft
    if not OPENAI_API_KEY or OpenAI is None:
        return draft_notarial_text(result.get("linderos", {}))

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        prompt = (
            "Redacta un párrafo notarial claro y conciso describiendo los linderos cardinales "
            "N/S/E/O con el formato típico en español. Si faltan lados, indícalo con '___'. "
            "No inventes datos. Devuelve una sola frase.\n\n"
            f"Datos:\n{json.dumps(result.get('linderos', {}), ensure_ascii=False)}"
        )
        # Using Chat Completions
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": "Eres un asistente notarial muy preciso."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        log.warning(f"OpenAI API error, using fallback: {e}")
        return draft_notarial_text(result.get("linderos", {}))

def save_txt_and_get_filename(text: str) -> str:
    ts = int(time.time())
    fname = f"{ts}.txt"
    fpath = os.path.join(DOWNLOAD_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(text)
    return fname

# ----------------------
# Routes
# ----------------------

@app.get("/", response_class=PlainTextResponse, include_in_schema=False)
def root():
    return "autoCata microservice online"

@app.get("/health")
def health():
    return {"status": "ok", "version": app.version}

# Public download (no auth)
@app.get("/download/{filename}")
def download_txt(filename: str):
    # Security: only allow .txt inside DOWNLOAD_DIR
    if not re.fullmatch(r"[A-Za-z0-9_\-]+\.txt", filename):
        raise HTTPException(status_code=400, detail="Nombre de archivo inválido")
    fpath = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(fpath):
        raise HTTPException(status_code=404, detail="Archivo no encontrado")
    return FileResponse(fpath, media_type="text/plain", filename=filename)

# Protected extract
@app.post("/extract")
async def extract_endpoint(
    file: UploadFile = File(...),
    credentials: dict = Depends(verify_jwt),
):
    if not file or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Debes subir un PDF catastral")

    pdf_bytes = await file.read()

    # OCR quick check
    # We'll allow the pipeline to decide strict rejection, but we also proactively check if bytes exist
    if len(pdf_bytes) < 2000:
        raise HTTPException(status_code=400, detail="El PDF parece estar corrupto o demasiado pequeño")

    # Run strict+fallback
    result, used_fallback = extract_with_strict_fallback(pdf_bytes)

    # En caso de OCR insuficiente, y si así se exige, rechazamos 400
    if REJECT_NO_OCR and (result.get("note") == "OCR insuficiente" or int(result.get("debug", {}).get("len_text", 0)) < MIN_OCR_CHARS):
        raise HTTPException(status_code=400, detail="El PDF no contiene texto OCR legible")

    # Si incluso tras fallback no hay suficiente
    if not is_valid_extraction(result, strict=False):
        raise HTTPException(status_code=400, detail="No se pudo extraer información suficiente ni en modo estricto ni en fallback.")

    # 1 sola llamada a GPT con el resultado final
    notarial_text = await generate_notarial_text_with_gpt(result)
    result["notarial_text"] = notarial_text

    # Guardar .txt y url pública
    txt_name = save_txt_and_get_filename(notarial_text)
    result["download_url"] = f"/download/{txt_name}"

    # debug info
    dbg = result.get("debug") or {}
    dbg["used_fallback"] = used_fallback
    result["debug"] = dbg

    return JSONResponse(result)

# ---------------
# Uvicorn entry
# ---------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

