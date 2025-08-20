from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import pdfplumber
from pdf2image import convert_from_bytes
import numpy as np
import cv2
from PIL import Image

app = FastAPI(title="AutoCatastro AI", version="0.3.0")

# ---------------- Flags / Auth ----------------
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "")
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")  # fuerza solo texto (sin imagen)
PAGE2_MAP  = (os.getenv("PAGE2_MAP",  "1").strip() == "1")  # usar mapa de pág. 2 (verde/rosa)

def check_token(x_autocata_token: str = Header(default="")):
    if AUTH_TOKEN and x_autocata_token != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")

# ---------------- Modelos ----------------
class ExtractIn(BaseModel):
    pdf_url: AnyHttpUrl

class ExtractOut(BaseModel):
    linderos: Dict[str, str]
    owners_detected: List[str] = []
    note: Optional[str] = None
    debug: Optional[dict] = None

# ---------------- Utilidades texto ----------------
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)
DNI_RE = re.compile(r"\b\d{8}[A-Z]\b")
PARCEL_ONLY_RE = re.compile(r"PARCELA\s+(\d{1,5})", re.IGNORECASE)

STOP_IN_NAME = (
    "POLÍGONO","POLIGONO","PARCELA","[","]","(",")",
    "COORDENADAS","ETRS","HUSO","ESCALA","TITULARIDAD",
    "VALOR CATASTRAL","LOCALIZACIÓN","LOCALIZACION"
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
    if not line: return False
    for bad in STOP_IN_NAME:
        if bad in line.upper(): return False
    if sum(ch.isdigit() for ch in line) >= 3: return False
    return bool(UPPER_NAME_RE.match(line))

def reconstruct_owner_from_block(lines: List[str]) -> str:
    tokens=[]
    for ln in lines:
        ln = re.sub(r"\s+"," ",ln.strip())
        if ln: tokens.extend(ln.split(" "))
    clean=[]; i=0
    while i<len(tokens):
        tok=re.sub(r"[^A-ZÁÉÍÓÚÜÑ\-\.'']", "", tokens[i])
        if not tok: i+=1; continue
        if len(tok)==1 and i+1<len(tokens):
            nxt=re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+1])
            if nxt and len(nxt)>=2: clean.append(tok+nxt); i+=2; continue
        if len(tok)<=2 and i+1<len(tokens):
            nxt=re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+1])
            if nxt and len(nxt)>=3: clean.append(tok+nxt); i+=2; continue
        if i+2<len(tokens):
            n1=re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+1])
            n2=re.sub(r"[^A-ZÁÉÍÓÚÜÑ]", "", tokens[i+2])
            if n1 and len(n1)==1 and n2 and len(n2)>=3:
                clean.append(tok+n1+n2); i+=3; continue
        clean.append(tok); i+=1
    name=" ".join(clean)
    return re.sub(r"\s{2,}"," ",name).strip()

def extract_owners_by_parcel(pdf_bytes: bytes) -> Dict[str,str]:
    """Devuelve dict: parcela -> titular (desde bloques de la certificación)."""
    mapping={}
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            lines = normalize_text(text).split("\n")
            i=0
            while i<len(lines):
                up=lines[i].upper()
                m=re.search(r"PARCELA\s+(\d{1,5})", up, re.I)
                if m:
                    parcela=m.group(1)
                    block=[]; j=i+1
                    while j<len(lines) and len(block)<4:
                        raw=lines[j].strip()
                        if not raw:
                            if block: break
                            j+=1; continue
                        if is_upper_name(raw):
                            block.append(raw); j+=1
                        else:
                            break
                    name = reconstruct_owner_from_block(block) if block else ""
                    if parcela and name and parcela not in mapping:
                        mapping[parcela]=name
                    i=j; continue
                i+=1
    return mapping

# ---------------- Visión (página 2) ----------------
def page2_to_bgr(pdf_bytes: bytes) -> np.ndarray:
    """Rasteriza la página 2 (si no existe, toma la 1)."""
    dpi = 300 if FAST_MODE else 450
    pages = convert_from_bytes(pdf_bytes, dpi=dpi)
    idx = 1 if len(pages)>=2 else 0
    pil: Image.Image = pages[idx].convert("RGB")
    return np.array(pil)[:,:,::-1]  # BGR

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
    """Devuelve (mask_verde_principal, mask_rosa_vecinos)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Verde agua de Catastro (dos rangos por variación)
    greens = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255,255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    # Rosa de colindantes (rojos/rosas en HSV)
    pinks = [
        (np.array([160, 25, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 25, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo,hi in greens: mg = cv2.bitwise_or(mg, cv2.inRange(hsv,lo,hi))
    for lo,hi in pinks:  mp = cv2.bitwise_or(mp, cv2.inRange(hsv,lo,hi))
    k3 = np.ones((3,3),np.uint8); k5 = np.ones((5,5),np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def biggest_centroid(mask: np.ndarray) -> Optional[Tuple[int,int]]:
    cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    c = max(cnts, key=cv2.contourArea)
    M = cv2.moments(c)
    if M["m00"]==0: return None
    return int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])

def touching_components(mask_ref: np.ndarray, mask_other: np.ndarray) -> List[np.ndarray]:
    """Devuelve contornos de 'other' que tocan (o casi) a 'ref'."""
    ref = cv2.dilate(mask_ref, np.ones((7,7),np.uint8), iterations=1)
    touch = cv2.bitwise_and(ref, mask_other)
    cnts,_ = cv2.findContours(mask_other, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    outs=[]
    for c in cnts:
        cmask = np.zeros_like(mask_other)
        cv2.drawContours(cmask, [c], -1, 255, thickness=cv2.FILLED)
        if cv2.countNonZero(cv2.bitwise_and(cmask, ref))>0:
            outs.append(c)
    return outs

def side_of(center_main: Tuple[int,int], pt: Tuple[int,int]) -> str:
    cx,cy = center_main
    vx,vy = pt[0]-cx, pt[1]-cy
    ang = math.degrees(math.atan2(-(vy), vx))  # Norte arriba
    if -45 <= ang <= 45:  return "este"
    if 45 < ang <= 135:   return "norte"
    if -135 <= ang < -45: return "sur"
    return "oeste"

# ---------- Texto vectorial (números de parcela en pág. 2) ----------
def digit_words_pdf(page) -> List[dict]:
    """Lista de 'palabras' que son números (texto vectorial) con su bbox en coords PDF."""
    words = page.extract_words() or []
    out=[]
    for w in words:
        txt = (w.get("text") or "").strip()
        if txt.isdigit():
            out.append({
                "text": txt,
                "x0": w["x0"], "top": w["top"], "x1": w["x1"], "bottom": w["bottom"]
            })
    return out

def pdfbox_to_imgpt(x: float, y: float, pdf_w: float, pdf_h: float, img_w: int, img_h: int) -> Tuple[int,int]:
    # PDF origen abajo-izq; imagen origen arriba-izq
    xi = int(x * img_w / pdf_w)
    yi = int((pdf_h - y) * img_h / pdf_h)
    return xi, yi

def parcel_in_component(words: List[dict], comp_cnt: np.ndarray,
                        pdf_w: float, pdf_h: float, img_w: int, img_h: int) -> Optional[str]:
    best = None
    for w in words:
        cx_pdf = (w["x0"] + w["x1"]) / 2.0
        cy_pdf = (w["top"] + w["bottom"]) / 2.0
        cx, cy = pdfbox_to_imgpt(cx_pdf, cy_pdf, pdf_w, pdf_h, img_w, img_h)
        inside = cv2.pointPolygonTest(comp_cnt, (float(cx), float(cy)), False)
        if inside >= 0:
            # elige el de mayor longitud (nº más “serio”)
            if best is None or len(w["text"]) > len(best):
                best = w["text"]
    return best

# ---------------- Endpoint principal ----------------
@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    # 0) Extrae mapeo parcela→titular desde el texto de la certificación
    parcel2owner = extract_owners_by_parcel(pdf_bytes)

    # Modo solo texto (emergencia/rápido)
    if TEXT_ONLY:
        note = "Modo TEXT_ONLY activo: mapa desactivado."
        owners_detected = list(dict.fromkeys(parcel2owner.values()))[:8]
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=owners_detected, note=note,
                          debug=({"TEXT_ONLY": True} if debug else None))

    # 1) Página 2: segmentación por color + lectura de números vectoriales
    if not PAGE2_MAP:
        return ExtractOut(linderos={"norte":"","sur":"","oeste":"","este":""},
                          owners_detected=list(dict.fromkeys(parcel2owner.values()))[:8],
                          note="PAGE2_MAP desactivado.", debug=None)

    # Rasteriza pág. 2
    bgr = page2_to_bgr(pdf_bytes)
    img_h, img_w = bgr.shape[:2]
    mask_g, mask_p = color_masks(bgr)
    main_center = biggest_centroid(mask_g)

    # Contornos rosas que tocan la verde
    comps = touching_components(mask_g, mask_p)

    # Palabras (números) en coords PDF
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_idx = 1 if len(pdf.pages)>=2 else 0
        page = pdf.pages[page_idx]
        pdf_w = float(page.width); pdf_h = float(page.height)
        words = digit_words_pdf(page)

    # Asignar a cada comp el nº de parcela
    comp_info = []  # (centro, parcela_num)
    for c in comps:
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"]/M["m00"]); cy = int(M["m01"]/M["m00"])
        num = parcel_in_component(words, c, pdf_w, pdf_h, img_w, img_h)
        if num: comp_info.append(((cx,cy), num))

    # 2) Cardinales desde el centro de la verde
    linderos = {"norte":"","sur":"","oeste":"","este":""}
    best_per_side: Dict[str, Tuple[str, Tuple[int,int], int]] = {}
    if main_center:
        for (cx,cy), num in comp_info:
            sd = side_of(main_center, (cx,cy))
            d2 = (cx-main_center[0])**2 + (cy-main_center[1])**2
            if sd not in best_per_side or d2 < best_per_side[sd][2]:
                best_per_side[sd] = (num, (cx,cy), d2)

    # 3) Convertir nº parcela → titular si está en el texto
    for sd, (num, _, _) in best_per_side.items():
        if num in parcel2owner:
            linderos[sd] = parcel2owner[num]

    owners_detected = list(dict.fromkeys(parcel2owner.values()))[:10]
    note = None
    if not any(linderos.values()):
        note = "No se obtuvieron titulares desde el mapa; revisaremos rangos de color o texto fuente."

    dbg = None
    if debug:
        dbg = {
            "FAST_MODE": FAST_MODE,
            "PAGE2_MAP": PAGE2_MAP,
            "img_wh": [img_w, img_h],
            "main_center": main_center,
            "comps": [{"center": c, "parcel": n} for (c,n) in comp_info],
            "best_per_side": {k: v[0] for k,v in best_per_side.items()}
        }

    return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

# ---------------- Preview (debug visual) ----------------
@app.get("/preview", dependencies=[Depends(check_token)])
def preview(pdf_url: AnyHttpUrl):
    """Devuelve PNG de la pág. 2 dibujando verde/rosa y números detectados."""
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    bgr = page2_to_bgr(pdf_bytes); img = bgr.copy()
    mg, mp = color_masks(bgr)
    cv2.drawContours(img, cv2.findContours(mg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0,255,0), 3)
    cv2.drawContours(img, cv2.findContours(mp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (180,105,255), 2)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        page_idx = 1 if len(pdf.pages)>=2 else 0
        page = pdf.pages[page_idx]
        pdf_w = float(page.width); pdf_h = float(page.height)
        words = digit_words_pdf(page)

    h,w = img.shape[:2]
    for wd in words:
        x,y = pdfbox_to_imgpt((wd["x0"]+wd["x1"])/2, (wd["top"]+wd["bottom"])/2, pdf_w, pdf_h, w, h)
        cv2.circle(img, (x,y), 7, (255,255,255), -1)
        cv2.putText(img, wd["text"], (x+6,y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    # PNG en respuesta
    import io as iio
    _, buf = cv2.imencode(".png", img)
    return app.response_class(buf.tobytes(), media_type="image/png")

# ---------------- Salud ----------------
@app.get("/health")
def health():
    return {"ok": True, "version": "0.3.0", "FAST_MODE": FAST_MODE, "TEXT_ONLY": TEXT_ONLY, "PAGE2_MAP": PAGE2_MAP}


