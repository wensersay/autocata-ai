from fastapi import FastAPI, HTTPException, Body, Depends, Header, Query
from pydantic import BaseModel, AnyHttpUrl
from starlette.responses import StreamingResponse
from typing import Dict, List, Optional, Tuple
import requests, io, re, os, math
import numpy as np
from pdf2image import convert_from_bytes
from PIL import Image
import cv2
import pytesseract

# ──────────────────────────────────────────────────────────────────────────────
# App & versión
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoCatastro AI", version="0.4.8")

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
# Utilidades comunes
# ──────────────────────────────────────────────────────────────────────────────
UPPER_NAME_RE = re.compile(r"^[A-ZÁÉÍÓÚÜÑ][A-ZÁÉÍÓÚÜÑ\s\.'\-]+$", re.UNICODE)

BAD_TOKENS = {
    "POLÍGONO","POLIGONO","PARCELA","APELLIDOS","NOMBRE","RAZON","RAZÓN",
    "SOCIAL","NIF","DOMICILIO","LOCALIZACIÓN","LOCALIZACION","REFERENCIA",
    "CATASTRAL","TITULARIDAD","PRINCIPAL","APELLIDOS NOMBRE","RAZON SOCIAL"
}

GEO_TOKENS = {
    "LUGO","BARCELONA","MADRID","VALENCIA","SEVILLA","CORUÑA","A CORUÑA",
    "MONFORTE","LEM","LEMOS","HOSPITALET","L'HOSPITALET","SAVIAO","SAVIÑAO",
    "GALICIA","[LUGO]","[BARCELONA]"
}

NAME_CONNECTORS = {"DE","DEL","LA","LOS","LAS","DA","DO","DAS","DOS","Y"}

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

def cv_flag(name: str, default: int = 0) -> int:
    return int(getattr(cv2, name, default))

THRESH_BINARY     = cv_flag("THRESH_BINARY", 0)
THRESH_BINARY_INV = cv_flag("THRESH_BINARY_INV", 0)
THRESH_OTSU       = cv_flag("THRESH_OTSU", 0)

# ──────────────────────────────────────────────────────────────────────────────
# Raster (pág. 2) y masks
# ──────────────────────────────────────────────────────────────────────────────
def page2_bgr(pdf_bytes: bytes) -> np.ndarray:
    dpi = 400 if FAST_MODE else 500
    pages = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=2, last_page=2)
    if not pages:
        raise HTTPException(status_code=400, detail="No se pudo rasterizar la página 2.")
    pil: Image.Image = pages[0].convert("RGB")
    return np.array(pil)[:, :, ::-1]

def color_masks(bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    g_ranges = [
        (np.array([35,  20, 50], np.uint8), np.array([85, 255, 255], np.uint8)),
        (np.array([86,  15, 50], np.uint8), np.array([100,255,255], np.uint8)),
    ]
    p_ranges = [
        (np.array([160, 20, 70], np.uint8), np.array([179,255,255], np.uint8)),
        (np.array([  0, 20, 70], np.uint8), np.array([ 10,255,255], np.uint8)),
    ]
    mg = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in g_ranges:
        mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges:
        mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
    k3 = np.ones((3,3), np.uint8); k5 = np.ones((5,5), np.uint8)
    mg = cv2.morphologyEx(mg, cv2.MORPH_OPEN, k3); mg = cv2.morphologyEx(mg, cv2.MORPH_CLOSE, k5)
    mp = cv2.morphologyEx(mp, cv2.MORPH_OPEN, k3); mp = cv2.morphologyEx(mp, cv2.MORPH_CLOSE, k5)
    return mg, mp

def contours_centroids(mask: np.ndarray, min_area: int) -> List[Tuple[int,int,int]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: List[Tuple[int,int,int]] = []
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area: continue
        M = cv2.moments(c)
        if M["m00"] == 0: continue
        cx = int(M["m10"] / M["m00"]); cy = int(M["m01"] / M["m00"])
        out.append((cx, cy, int(a)))
    out.sort(key=lambda x: -x[2])
    return out

def side_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = math.degrees(math.atan2(-(sy), sx))
    if -45 <= ang <= 45: return "este"
    if 45 < ang <= 135:  return "norte"
    if -135 <= ang < -45:return "sur"
    return "oeste"

# ──────────────────────────────────────────────────────────────────────────────
# OCR utils
# ──────────────────────────────────────────────────────────────────────────────
def enhance_gray(g: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(g)

def binarize(gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    flags_bin = THRESH_BINARY | (THRESH_OTSU if THRESH_OTSU else 0)
    flags_inv = THRESH_BINARY_INV | (THRESH_OTSU if THRESH_OTSU else 0)
    _, bw  = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_bin)
    _, bwi = cv2.threshold(gray, 0 if THRESH_OTSU else 127, 255, flags_inv)
    return bw, bwi

def ocr_text(img: np.ndarray, psm: int, whitelist: Optional[str] = None) -> str:
    cfg = f"--psm {psm} --oem 3"
    if whitelist is not None:
        safe = (whitelist or "").replace('"','')
        cfg += f' -c tessedit_char_whitelist="{safe}"'
    txt = pytesseract.image_to_string(img, config=cfg) or ""
    txt = txt.replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()

def clean_owner_line(line: str) -> str:
    if not line: return ""
    toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", line.upper()) if t]
    out = []
    for t in toks:
        if any(ch.isdigit() for ch in t): break
        if t in GEO_TOKENS or "[" in t or "]" in t: break
        if t in BAD_TOKENS: continue
        out.append(t)
        if len([x for x in out if x not in NAME_CONNECTORS]) >= 5:
            break
    compact = []
    for t in out:
        if (not compact) and t in NAME_CONNECTORS:
            continue
        compact.append(t)
    name = " ".join(compact).strip()
    return name[:48]

def pick_owner_from_text(txt: str) -> str:
    if not txt: return ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines:
        U = l.upper()
        if any(tok in U for tok in BAD_TOKENS):   continue
        if sum(ch.isdigit() for ch in U) > 1:     continue
        if not UPPER_NAME_RE.match(U):            continue
        name = clean_owner_line(U)
        if len(name) >= 8:
            return name
    return ""

def join_name_lines(line1: str, line2: str) -> str:
    """Une dos líneas de nombre si la 2ª parece continuación válida."""
    l1 = clean_owner_line(line1 or "")
    l2 = clean_owner_line(line2 or "")
    if not l1 and not l2: return ""
    if l1 and not l2:     return l1
    if l2 and not l1:     return l2
    # Evita repetir si l2 ya está incluido o es muy corta
    if l2 in l1 or len(l2) < 3: return l1
    # No unir si l2 parece dirección/lugar (heurística ya filtra mucho)
    return (l1 + " " + l2).strip()

# ──────────────────────────────────────────────────────────────────────────────
# Localizar columna de “Apellidos…” y extraer DOS líneas del titular
# ──────────────────────────────────────────────────────────────────────────────
def find_header_and_owner_bands(bgr: np.ndarray, row_y: int,
                                x_text0: int, x_text1: int) -> Tuple[int,int,int,int,int,int]:
    """
    Devuelve (x0, x1, y0_line1, y1_line1, y0_line2, y1_line2) para el bloque
    bajo la cabecera “Apellidos Nombre / Razón social”.
    """
    h, w = bgr.shape[:2]
    pad_y = int(h * 0.06)
    y0s = max(0, row_y - pad_y)
    y1s = min(h, row_y + pad_y)

    band = bgr[y0s:y1s, x_text0:x_text1]
    # line_height ~ una línea de texto
    line_h = max(18, int(h * 0.035))

    if band.size == 0:
        y0 = max(0, row_y - int(h*0.01))
        y1 = min(h, y0 + line_h)
        y2 = min(h, y1 + 2)
        y3 = min(h, y2 + line_h)
        x0 = x_text0
        x1 = int(x_text0 + 0.55*(x_text1-x_text0))
        return x0, x1, y0, y1, y2, y3

    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
    bw, bwi = binarize(gray)

    for im in (bw, bwi):
        data = pytesseract.image_to_data(im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3")
        words = data.get("text", [])
        xs = data.get("left", [])
        ys = data.get("top", [])
        ws = data.get("width", [])
        hs = data.get("height", [])

        x_nif = None
        header_bottom = None

        for t, lx, ty, ww, hh in zip(words, xs, ys, ws, hs):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T or "APELLIDOS NOMBRE" in T or "RAZON" in T or "RAZÓN" in T:
                header_bottom = max(header_bottom or 0, ty + hh)
            if T == "NIF":
                x_nif = lx
                header_bottom = max(header_bottom or 0, ty + hh)

        if header_bottom is not None:
            y0_line1 = y0s + header_bottom + 6
            y1_line1 = min(h, y0_line1 + line_h)
            y0_line2 = min(h, y1_line1 + 2)
            y1_line2 = min(h, y0_line2 + line_h)
            if x_nif is not None:
                x0 = x_text0
                x1 = min(x_text1, x_text0 + x_nif - 8)
            else:
                x0 = x_text0
                x1 = int(x_text0 + 0.55*(x_text1-x_text0))
            return x0, x1, y0_line1, y1_line1, y0_line2, y1_line2

    # Fallback conservador
    y0 = max(0, row_y - int(h*0.01))
    y1 = min(h, y0 + line_h)
    y2 = min(h, y1 + 2)
    y3 = min(h, y2 + line_h)
    x0 = x_text0
    x1 = int(x_text0 + 0.55*(x_text1-x_text0))
    return x0, x1, y0, y1, y2, y3

def extract_owner_from_row(bgr: np.ndarray, row_y: int) -> Tuple[str, dict, int]:
    """
    Devuelve (owner_joined, debug_dict, attempt_used).
    Hace hasta 3 intentos, expandiendo x0 a la izquierda si muerde letras.
    En cada intento lee 2 líneas y las une si procede.
    """
    h, w = bgr.shape[:2]
    x_text0 = int(w * 0.33)
    x_text1 = int(w * 0.96)

    best = ("", {"roi1":None,"roi2":None,"txt1":"","txt2":""}, -1)

    for attempt in range(3):
        extra_left = attempt * max(18, int(w * 0.03))
        x0_base = max(0, x_text0 - extra_left)
        x0, x1, y01, y11, y02, y12 = find_header_and_owner_bands(bgr, row_y, x0_base, x_text1)

        roi1 = bgr[y01:y11, x0:x1]
        roi2 = bgr[y02:y12, x0:x1]

        owner1 = owner2 = ""
        txt1 = txt2 = ""

        if roi1.size:
            g1 = cv2.cvtColor(roi1, cv2.COLOR_BGR2GRAY)
            g1 = cv2.resize(g1, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            g1 = enhance_gray(g1)
            bw1, bwi1 = binarize(g1)
            WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
            cands1 = [
                ocr_text(bw1,  psm=7, whitelist=WL),
                ocr_text(bwi1, psm=7, whitelist=WL),
                ocr_text(bw1,  psm=6, whitelist=WL),
                ocr_text(bwi1, psm=6, whitelist=WL),
            ]
            for c in cands1:
                txt1 = c
                owner1 = pick_owner_from_text(c)
                if owner1: break

        if roi2.size:
            g2 = cv2.cvtColor(roi2, cv2.COLOR_BGR2GRAY)
            g2 = cv2.resize(g2, None, fx=1.35, fy=1.35, interpolation=cv2.INTER_CUBIC)
            g2 = enhance_gray(g2)
            bw2, bwi2 = binarize(g2)
            WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
            cands2 = [
                ocr_text(bw2,  psm=7, whitelist=WL),
                ocr_text(bwi2, psm=7, whitelist=WL),
                ocr_text(bw2,  psm=6, whitelist=WL),
                ocr_text(bwi2, psm=6, whitelist=WL),
            ]
            for c in cands2:
                txt2 = c
                owner2 = pick_owner_from_text(c)
                if owner2: break

        joined = join_name_lines(owner1, owner2)
        if len(joined) > len(best[0]):
            best = (joined,
                    {"roi1":[x0,y01,x1,y11], "roi2":[x0,y02,x1,y12], "txt1":txt1, "txt2":txt2},
                    attempt)

        if joined and len(joined) >= 10:
            return joined, best[1], attempt

    return best[0], best[1], best[2]

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas (con anotaciones opcionales)
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # Recorte donde viven croquis + letras
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(360 if FAST_MODE else 240))
    neighs = contours_centroids(mp, min_area=(260 if FAST_MODE else 180))
    if not mains:
        return {"norte":"","sur":"","este":"","oeste":""}, {"rows": []}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    rows_dbg = []
    linderos = {"norte":"","sur":"","este":"","oeste":""}
    used_sides = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        # vecino más cercano en la fila
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.25)**2:
            side = side_of((mcx, mcy), best)

        owner_joined, owner_dbg, attempt_id = extract_owner_from_row(bgr, row_y=mcy)

        if side and owner_joined and side not in used_sides:
            linderos[side] = owner_joined
            used_sides.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl = {"norte":"N","sur":"S","este":"E","oeste":"O"}.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            # dibuja bandas ROI de línea1 y línea2
            if owner_dbg.get("roi1"):
                x0,y0,x1,y1 = owner_dbg["roi1"]
                cv2.rectangle(vis, (x0,y0), (x1,y1), (0,255,255), 2)
            if owner_dbg.get("roi2"):
                x0,y0,x1,y1 = owner_dbg["roi2"]
                cv2.rectangle(vis, (x0,y0), (x1,y1), (0,200,200), 2)
            if annotate_names and owner_joined:
                cv2.putText(vis, owner_joined[:28], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(vis, owner_joined[:28], (int(w*0.42), mcy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner_joined,
            "roi1": owner_dbg.get("roi1"),
            "roi2": owner_dbg.get("roi2"),
            "txt1": owner_dbg.get("txt1",""),
            "txt2": owner_dbg.get("txt2",""),
            "attempt": attempt_id
        })

    dbg = {"rows": rows_dbg}
    return linderos, dbg, vis

# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "ok": True,
        "version": app.version,
        "FAST_MODE": FAST_MODE,
        "TEXT_ONLY": TEXT_ONLY,
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)}
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(1, description="1=mostrar N/S/E/O"),
    names: int = Query(0, description="1=mostrar nombre estimado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((240, 640, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:60]}", (10,120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...),
                 labels: int = Query(1),
                 names: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note="Modo TEXT_ONLY activo: mapa/OCR desactivados.",
            debug={"TEXT_ONLY": True} if debug else None
        )

    try:
        bgr = page2_bgr(pdf_bytes)
        linderos, vdbg, _vis = detect_rows_and_extract(bgr, annotate=False)
        owners_detected = [o["owner"] for o in vdbg["rows"] if o.get("owner")]
        owners_detected = list(dict.fromkeys(owners_detected))[:8]

        note = None
        if not any(linderos.values()):
            note = "No se pudo determinar lado/vecino con suficiente confianza."

        dbg = vdbg if debug else None
        return ExtractOut(linderos=linderos, owners_detected=owners_detected, note=note, debug=dbg)

    except Exception as e:
        return ExtractOut(
            linderos={"norte":"","sur":"","este":"","oeste":""},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )

