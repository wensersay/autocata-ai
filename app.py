from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional
import requests, io, re
import pdfplumber

app = FastAPI(title="AutoCatastro AI", version="0.1.0")

# ---------- Modelos ----------
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None

# ---------- Utilidades ----------
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")

STOP_IN_NAME = (
    "POLÍGONO", "POLIGONO", "PARCELA", "[", "]", "(", ")",
    "COORDENADAS", "ETRS", "HUSO", "ESCALA", "TITULARIDAD",
    "VALOR CATASTRAL", "LOCALIZACIÓN", "LOCALIZACION"
)

def fetch_pdf_bytes(url: str) -> bytes:
    """Descarga el PDF y valida que realmente lo sea."""
    try:
        r = requests.get(url, timeout=40)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el PDF (HTTP {r.status_code}).")
        ct = (r.headers.get("content-type") or "").lower()
        if "pdf" not in ct:
            # fallback: comprueba firma %PDF
            if not r.content.startswith(b"%PDF"):
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
    for bad in STOP_IN_NAME:
        if bad in line.upper():
            return False
    if sum(ch.isdigit() for ch in line) >= 3:
        return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner_from_block(lines: List[str]) -> str:
    tokens = []
    for ln in lines:
        ln = re.sub(r"\s+", " ", ln.strip())
        if ln:
            tokens.extend(ln.split(" "))

    clean = []
    i = 0
    while i < len(tokens):
        tok = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", tokens[i])
        if not tok:
            i += 1
            continue

        # 1 letra → unir con siguiente largo
        if len(tok) == 1 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 2:
                clean.append(tok + nxt); i += 2; continue

        # 2 letras → unir si el siguiente es largo
        if len(tok) <= 2 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 3:
                clean.append(tok + nxt); i += 2; continue

        # patrón 'AL' + 'VAREZ'
        if i + 2 < len(tokens):
            nxt1 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            nxt2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 2])
            if nxt1 and len(nxt1) == 1 and nxt2 and len(nxt2) >= 3:
                clean.append(tok + nxt1 + nxt2); i += 3; continue

        clean.append(tok); i += 1

    name = " ".join(clean)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_owners_ordered(pdf_bytes: bytes) -> List[str]:
    """
    Heurística v1.1:
    A) Buscar NIF y subir ~12 líneas para reconstruir el/los NOMBRE(S).
    B) Fallback: tras 'Polígono ... Parcela ...', tomar 1–3 líneas MAYÚSCULAS como nombre.
    Devuelve hasta 4 nombres distintos, en orden de hallazgo.
    """
    owners: List[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # ---- Método A: NIF hacia arriba ----
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")

            for i, ln in enumerate(lines):
                if DNI_RE.search(ln):
                    prev_block = []
                    has_poly = any(
                        ("POLÍGONO" in lines[k].upper() or "POLIGONO" in lines[k].upper() or "PARCELA" in lines[k].upper())
                        for k in range(max(0, i - 12), i)
                    )
                    for j in range(i - 1, max(0, i - 12) - 1, -1):
                        raw = lines[j].strip()
                        if not raw:
                            continue
                        if ("POLÍGONO" in raw.upper() or "POLIGONO" in raw.upper() or "PARCELA" in raw.upper() or "[" in raw):
                            break
                        if not is_upper_name(raw):
                            break
                        prev_block.append(raw)
                    if not prev_block and not has_poly:
                        continue

                    prev_block.reverse()
                    name = reconstruct_owner_from_block(prev_block) if prev_block else ""
                    if name and name not in owners:
                        owners.append(name)
                    if len(owners) >= 4:
                        return owners

        # ---- Método B: desde 'Polígono/Parcela' hacia abajo ----
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")

            for i, ln in enumerate(lines):
                if ("POLÍGONO" in ln.upper() or "POLIGONO" in ln.upper()) and "PARCELA" in ln.upper():
                    block = []
                    for j in range(i + 1, min(i + 11, len(lines))):
                        raw = lines[j].strip()
                        if not raw:
                            if block:
                                name = reconstruct_owner_from_block(block)
                                if name and name not in owners:
                                    owners.append(name)
                                block = []
                            continue

                        if is_upper_name(raw):
                            block.append(raw)
                            if len(block) >= 3:
                                name = reconstruct_owner_from_block(block)
                                if name and name not in owners:
                                    owners.append(name)
                                block = []
                        else:
                            if block:
                                name = reconstruct_owner_from_block(block)
                                if name and name not in owners:
                                    owners.append(name)
                                block = []

                        if len(owners) >= 4:
                            return owners

                    if block and len(owners) < 4:
                        name = reconstruct_owner_from_block(block)
                        if name and name not in owners:
                            owners.append(name)
                        if len(owners) >= 4:
                            return owners

    return owners[:4]

# ---------- Endpoints ----------
@app.post("/extract", response_model=ExtractOut)
def extract(data: ExtractIn = Body(...)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))
    owners = extract_owners_ordered(pdf_bytes)

    linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}
    if len(owners) > 0: linderos["norte"] = owners[0]
    if len(owners) > 1: linderos["sur"]   = owners[1]
    if len(owners) > 2: linderos["oeste"] = owners[2]
    if len(owners) > 3: linderos["este"]  = owners[3]

    note = None
    if not owners:
        note = "No se detectaron titulares en el PDF con la heurística base. Luego añadiremos visión por computador/OCR."

    return ExtractOut(linderos=linderos, owners_detected=owners, note=note)

@app.get("/health")
def health():
    return {"ok": True}
