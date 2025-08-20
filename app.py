from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional
import os, io, re, requests, pdfplumber

app = FastAPI(title="AutoCatastro AI", version="0.2.5-SAFE")

# -------- Flags/entorno --------
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "1").strip() == "1")  # por defecto TRUE para evitar 500

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ----------------- Modelos -----------------
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None

# ----------------- Utilidades texto -----------------
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE        = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE= re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO","POLIGONO","PARCELA","[","]","(",")","COORDENADAS",
    "ETRS","HUSO","ESCALA","TITULARIDAD","VALOR CATASTRAL",
    "LOCALIZACIÓN","LOCALIZACION"
)

def fetch_pdf_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=60)
    if r.status_code != 200:
        raise Exception(f"No se pudo descargar el PDF (HTTP {r.status_code}).")
    if b"%PDF" not in r.content[:1024]:
        raise Exception("La URL no parece entregar un PDF válido.")
    return r.content

def normalize_text(s: str) -> str:
    s = s.replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{2,}", "\n", s)
    return s.strip()

def is_upper_name(line: str) -> bool:
    line = line.strip()
    if not line: return False
    for bad in STOP_IN_NAME:
        if bad in line.upper(): return False
    if sum(ch.isdigit() for ch in line) >= 3: return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner_from_block(lines: List[str]) -> str:
    tokens=[]
    for ln in lines:
        ln=re.sub(r"\s+"," ",ln.strip())
        if ln: tokens.extend(ln.split(" "))
    clean=[]; i=0
    while i < len(tokens):
        tok = re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", tokens[i])
        if not tok: i+=1; continue
        if len(tok)==1 and i+1<len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+1])
            if nxt and len(nxt)>=2: clean.append(tok+nxt); i+=2; continue
        if len(tok)<=2 and i+1<len(tokens):
            nxt = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+1])
            if nxt and len(nxt)>=3: clean.append(tok+nxt); i+=2; continue
        if i+2<len(tokens):
            n1 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+1])
            n2 = re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+2])
            if n1 and len(n1)==1 and n2 and len(n2)>=3:
                clean.append(tok+n1+n2); i+=3; continue
        clean.append(tok); i+=1
    name=" ".join(clean)
    return re.sub(r"\s{2,}"," ",name).strip()

def extract_owners_by_parcel(pdf_bytes: bytes) -> Dict[str,str]:
    mapping={}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        # prioriza pág. 2 en adelante
        pages = list(range(1, len(pdf.pages))) + [0]
        for idx in pages:
            text = normalize_text(pdf.pages[idx].extract_text(x_tolerance=2, y_tolerance=2) or "")
            lines = text.split("\n"); i=0
            while i < len(lines):
                lu = lines[i].upper()
                m = PARCEL_ONLY_RE.search(lu)
                if m:
                    parcel = m.group(1)
                    block=[]; j=i+1
                    while j<len(lines) and len(block)<4:
                        raw=lines[j].strip()
                        if not raw:
                            if block: break
                            j+=1; continue
                        if is_upper_name(raw): block.append(raw); j+=1
                        else: break
                    name = reconstruct_owner_from_block(block) if block else ""
                    if parcel and name and parcel not in mapping:
                        mapping[parcel]=name
                    i=j; continue
                i+=1
    return mapping

def extract_fallback_names(pdf_bytes: bytes) -> List[str]:
    owners=[]
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            for i,ln in enumerate(lines):
                if DNI_RE.search(ln):
                    prev=[]
                    for j in range(i-1, max(0,i-14)-1, -1):
                        raw=lines[j].strip()
                        if not raw: continue
                        if ("POLÍGONO" in raw.upper() or "POLIGONO" in raw.upper() or "PARCELA" in raw.upper() or "[" in raw):
                            break
                        if not is_upper_name(raw): break
                        prev.append(raw)
                    prev.reverse()
                    name = reconstruct_owner_from_block(prev) if prev else ""
                    if name and name not in owners:
                        owners.append(name)
                        if len(owners)>=4: return owners
    return owners

# ----------------- Endpoint -----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    # Nunca devolvemos 500: todo envuelto en try/except
    try:
        pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))
    except Exception as e:
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=[], note=f"Error obteniendo PDF: {e}")

    try:
        parcel2owner = extract_owners_by_parcel(pdf_bytes)
        fallback     = extract_fallback_names(pdf_bytes)

        # Modo solo-texto (por defecto) — estable
        linderos = {"norte":"","sur":"","oeste":"","este":""}
        used=set(); order=["norte","sur","oeste","este"]; i=0
        for side in order:
            while i < len(fallback) and fallback[i] in used: i+=1
            if i < len(fallback):
                linderos[side]=fallback[i]; used.add(fallback[i]); i+=1

        owners_detected = list(dict.fromkeys(list(parcel2owner.values()) + fallback))[:8]
        note = "Modo TEXT_ONLY activo: OCR desactivado para evitar errores internos."

        dbg = None
        if debug:
            # pequeño vistazo para depurar
            dbg = {"pages_hint":"pág. 2 priorizada", "owners_by_parcel_sample": dict(list(parcel2owner.items())[:6])}

        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        # Cualquier fallo interno → respuesta segura
        dbg = {"exception": str(e)} if debug else None
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=[], note=f"Excepción en análisis: {e}", debug=dbg)

@app.get("/health")
def health():
    return {"ok": True, "version": "0.2.5-SAFE", "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY}


