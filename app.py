from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional
import requests, io, re
import pdfplumber

app = FastAPI(title="AutoCatastro AI", version="0.1.0")

class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None

UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")

def fetch_pdf_bytes(url: str) -> bytes:
    try:
        r = requests.get(url, timeout=40)
        if r.status_code != 200:
            raise HTTPException(status_code=400, detail=f"No se pudo descargar el PDF (HTTP {r.status_code}).")
        ct = r.headers.get("content-type", "")
        if "pdf" not in ct.lower():
            # A veces servidores no ponen el content-type correcto; seguimos si los bytes parecen PDF
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
    if not line: return False
    if "Polígono" in line or "Poligono" in line or "Parcela" in line:
        return False
    if re.search(r"\d", line):  # fuera números
        return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner_from_block(lines: List[str]) -> str:
    """
    Une tokens de líneas en MAYÚSCULAS para reconstruir nombres partidos
    (p.ej., 'RODRIGUEZ AL V AREZ JOSE LUIS').
    """
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
                clean.append(tok + nxt)
                i += 2
                continue

        # 2 letras + largo
        if len(tok) <= 2 and i + 1 < len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            if nxt and len(nxt) >= 3:
                clean.append(tok + nxt)
                i += 2
                continue

        # patrón ... 'AL' 'VAREZ'
        if i + 2 < len(tokens):
            nxt1 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 1])
            nxt2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i + 2])
            if nxt1 and len(nxt1) == 1 and nxt2 and len(nxt2) >= 3:
                clean.append(tok + nxt1 + nxt2)
                i += 3
                continue

        clean.append(tok)
        i += 1

    name = " ".join(clean)
    name = re.sub(r"\s{2,}", " ", name).strip()
    return name

def extract_owners_ordered(pdf_bytes: bytes) -> List[str]:
    owners: List[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages):
            text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
            text = normalize_text(text)
            lines = text.split("\n")

            for i, ln in enumerate(lines):
                if DNI_RE.search(ln):
                    has_poly = any(("Polígono" in lines[k] or "Poligono" in lines[k] or "Parcela" in lines[k])
                                   for k in range(max(0, i-8), i))
                    if not has_poly:
                        continue

                    prev_block = []
                    for j in range(i - 1, max(0, i - 7) - 1, -1):
                        raw = lines[j].strip()
                        if not raw:
                            continue
                        if ("Polígono" in raw) or ("Poligono" in raw) or ("Parcela" in raw) or ("[" in raw):
                            break
                        if not is_upper_name(raw):
                            break
                        prev_block.append(raw)
                    if not prev_block:
                        continue

                    prev_block.reverse()
                    name = reconstruct_owner_from_block(prev_block)
                    if name and name not in owners:
                        owners.append(name)
                    if len(owners) >= 4:
                        return owners
    return owners

from fastapi import Body

@app.post("/extract")
def extract(data: ExtractIn = Body(...)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))
    owners = extract_owners_ordered(pdf_bytes)

    linderos = {"norte": "", "sur": "", "oeste": "", "este": ""}
    if owners:
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
