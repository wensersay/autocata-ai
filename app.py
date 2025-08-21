from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import numpy as np
import pdfplumber
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.3.2")

# ──────────────────────────────────────────────────────────────────────────────
# Flags de entorno / seguridad
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ──────────────────────────────────────────────────────────────────────────────
# Modelos I/O
# ──────────────────────────────────────────────────────────────────────────────
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None

# ──────────────────────────────────────────────────────────────────────────────
# Utilidades PDF / Texto
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE        = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE= re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION", "REFERENCIA"
)

def fetch_pdf_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el PDF (HTTP {r.status_code}).")
        ct = (r.headers.get("content-type") or "").lower()
        if "pdf" not in ct and not r.content.startswith(b"%PDF"):
            raise HTTPException(status_code=400, detail="La URL no parece entregar un PDF válido.")
        return r.content
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error de red al descargar el PDF: {e}")

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def is_upper_name(line: str) -> bool:
    line = line.strip()
    if not line:
        return False
    U = line.upper()
    for bad in STOP_IN_NAME:
        if bad in U:
            return False
    if sum(ch.isdigit() for ch in line) >= 3:
        return False
    return bool(UPPER_NAME_RE.match(line))

def looks_location(line: str) -> bool:
    """Heurística para descartar la línea de localización (MAROUZAS... [LUGO])"""
    U = line.upper().strip()
    if "[" in U and "]" in U:
        return True
    if "." in line:  # MAROUZAS. O SAVIÑAO ...
        return True
    loc_terms = (" LUGO", " PONTEVEDRA", " CORUÑA", " CORUNA", " OURENSE", " MADRID",
                 " BARCELONA", " ASTURIAS", " SEVILLA", " VALENCIA", " BIZKAIA",
                 " GIPUZKOA", " ALAVA", " CANTABRIA", " MURCIA", " ALICANTE",
                 " CASTELLON", " GRANADA")
    return any(t in U for t in loc_terms)

def clean_name_prefix(line: str) -> str:
    """Devuelve la parte anterior al primer dígito (corta NIF/direcciones en la misma línea)."""
    part = re.split(r"\d", line, 1)[0].strip()
    part = re.sub(r"\s{2,}", " ", part)
    return part

def reconstruct_owner(lines: List[str]) -> str:
    """Une 1–3 líneas en mayúsculas en un nombre razonable (incluye segundas líneas tipo 'LUIS')."""
    toks: List[str] = []
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln.strip())
        if ln:
            toks.extend(ln.split(" "))
    clean = []
    i = 0
    while i < len(toks):
        tok = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", toks[i])
        if not tok:
            i += 1
            continue
        # pegados típicos (A L VARELA → ALVARELA)
        if len(tok) <= 2 and i + 1 < len(toks):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", toks[i+1])
            if nxt and len(nxt) >= 3:
                clean.append(tok + nxt)
                i += 2
                continue
        clean.append(tok)
        i += 1
    name = " ".join(clean)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_owners_map(pdf_bytes: bytes) -> Dict[str, str]:
    """
    Lee páginas ≥2 y construye { parcela: titular }.
    - Salta la línea de localización (MAROUZAS... [LUGO]).
    - Extrae el nombre aunque comparta renglón con el NIF (se queda con lo previo al primer dígito).
    - Si justo debajo hay un renglón con un único nombre en mayúsculas (p.ej. 'LUIS'), lo añade.
    """
    mapping: Dict[str, str] = {}

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for pi, page in enumerate(pdf.pages):
            if pi == 0:
                continue  # saltar portada

            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")

            curr_parcel: Optional[str] = None
            i = 0
            while i < len(lines):
                raw = lines[i].strip()
                up  = raw.upper()

                # (1) Detectar "PARCELA N"
                if "PARCELA" in up:
                    tokens = [t for t in up.replace(",", " ").split() if t.isdigit()]
                    if tokens:
                        curr_parcel = tokens[-1]
                    i += 1
                    # Posible línea de localización inmediatamente después → sáltala
                    if i < len(lines) and looks_location(lines[i]):
                        i += 1
                    continue

                # (2) Bloque de titularidad (tabla)
                if ("TITULARIDAD PRINCIPAL" in up) or ("APELLIDOS NOMBRE" in up and "RAZON" in up):
                    j = i + 1

                    def is_meta(s: str) -> bool:
                        U = s.upper().strip()
                        return (U == "" or "APELLIDOS NOMBRE" in U or "RAZON SOCIAL" in U or
                                "NIF" in U or "DOMIC

