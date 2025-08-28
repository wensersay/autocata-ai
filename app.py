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
app = FastAPI(title="AutoCatastro AI", version="0.6.1")

# ──────────────────────────────────────────────────────────────────────────────
# Flags / entorno
# ──────────────────────────────────────────────────────────────────────────────
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "").strip()
FAST_MODE  = (os.getenv("FAST_MODE",  "1").strip() == "1")
TEXT_ONLY  = (os.getenv("TEXT_ONLY",  "0").strip() == "1")

def get_int_env(name: str, default: int) -> int:
    try:
        v = int(os.getenv(name, "").strip() or default)
        return max(72, min(800, v))
    except Exception:
        return default

def get_dpi() -> int:
    if FAST_MODE:
        return get_int_env("FAST_DPI", 340)  # manda en FAST_MODE
    return get_int_env("PDF_DPI", 420)

# Mini-lista de nombres frecuentes + ampliable por ENV
NAME_HINTS = {
    "JOSE","LUIS","MARIA","CARMEN","ANA","ANTONIO","JUAN","FRANCISCO",
    "MANUEL","JAVIER","MIGUEL","RAFAEL","PEDRO","ALVARO","JORGE","MARTA",
    "ROBERTO","FERNANDO","PABLO","ALFONSO","IGNACIO","DIEGO","LAURA",
}
EXTRA = [t.strip().upper() for t in (os.getenv("NAME_HINTS_EXTRA","").split(",")) if t.strip()]
NAME_HINTS.update(EXTRA)

# Tokens basura para 2ª línea (ENV: JUNK_2NDLINE="Z,VA,EO,SS,KO,KR")
JUNK_2NDLINE = {t.strip().upper() for t in os.getenv("JUNK_2NDLINE","").split(",") if t.strip()}

# ──────────────────────────────────────────────────────────────────────────────
# Seguridad
# ──────────────────────────────────────────────────────────────────────────────
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
    "CATASTRAL","TITULARIDAD","PRINCIPAL"
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
    dpi = get_dpi()
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
    for lo, hi in g_ranges: mg = cv2.bitwise_or(mg, cv2.inRange(hsv, lo, hi))
    mp = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in p_ranges: mp = cv2.bitwise_or(mp, cv2.inRange(hsv, lo, hi))
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

# 8 rumbos
def side8_of(main_xy: Tuple[int,int], pt_xy: Tuple[int,int]) -> str:
    cx, cy = main_xy
    x, y   = pt_xy
    sx, sy = x - cx, y - cy
    ang = (math.degrees(math.atan2(-(sy), sx)) + 360.0) % 360.0  # 0=E, 90=N
    # centros de bins cada 45º: E=0, NE=45, N=90, ...
    labels = ["este","noreste","norte","noroeste","oeste","suroeste","sur","sureste"]
    idx = int(((ang + 22.5) % 360) // 45)
    return labels[idx]

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
        if any(tok in U for tok in BAD_TOKENS):
            continue
        if sum(ch.isdigit() for ch in U) > 1:
            continue
        if not UPPER_NAME_RE.match(U):
            continue
        name = clean_owner_line(U)
        if len(name) >= 6:
            return name
    return ""

# ──────────────────────────────────────────────────────────────────────────────
# Ubicación de columna (una sola vez por página)
# ──────────────────────────────────────────────────────────────────────────────
def find_columns_once(bgr: np.ndarray) -> dict:
    h, w = bgr.shape[:2]
    x_name0 = int(w * 0.33)
    x_name1 = int(w * 0.60)
    header_left = None
    x_nif = None

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = enhance_gray(gray)
    bw, bwi = binarize(gray)

    for im in (bw, bwi):
        try:
            data = pytesseract.image_to_data(
                im, output_type=pytesseract.Output.DICT, config="--psm 6 --oem 3"
            )
        except Exception:
            continue

        words = data.get("text", [])
        xs = data.get("left", [])

        for i, t in enumerate(words):
            if not t: continue
            T = t.upper()
            if "APELLIDOS" in T and header_left is None:
                header_left = int(xs[i])
            if T == "NIF" and x_nif is None:
                x_nif = int(xs[i])

        if header_left is not None or x_nif is not None:
            break

    if header_left is not None:
        x_name0 = max(0, header_left - 8)
    if x_nif is not None:
        x_name1 = max(x_name0 + 40, min(w - 1, x_nif - 8))

    return {
        "x0": int(x_name0),
        "x1": int(x_name1),
        "header_left_abs": int(header_left) if header_left is not None else None,
        "x_nif_abs": int(x_nif) if x_nif is not None else None,
    }

# ──────────────────────────────────────────────────────────────────────────────
# Extracción por fila (1ª y 2ª línea)
# ──────────────────────────────────────────────────────────────────────────────
def extract_owner_from_row(bgr: np.ndarray, row_y: int, cols: dict) -> Tuple[str, dict]:
    """
    Devuelve (owner, dbg)
    - cols: dict con x0/x1 y marcas (find_columns_once)
    - Lee 1ª línea + (opcional) 2ª línea; si la 1ª trae salto de línea, usa ese resto.
    """
    h, w = bgr.shape[:2]
    x0, x1 = cols.get("x0", int(w*0.33)), cols.get("x1", int(w*0.60))

    # Altura por línea (escala con la página)
    line_h = max(22, int(h * 0.035))
    y1a = max(0, row_y - int(h * 0.01))
    y1b = min(h, y1a + line_h)
    y2a = min(h, y1b + 2)
    y2b = min(h, y2a + line_h)

    # ROI y OCR
    band1 = bgr[y1a:y1b, x0:x1]
    band2 = bgr[y2a:y2b, x0:x1]

    def ocr_upper(img: np.ndarray) -> str:
        if img.size == 0: return ""
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, None, fx=1.25, fy=1.25, interpolation=cv2.INTER_CUBIC)
        g = enhance_gray(g)
        bw, bwi = binarize(g)
        WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZÁÉÍÓÚÜÑ '"
        txts = [
            ocr_text(bw, 6, WL),
            ocr_text(bwi, 6, WL),
            ocr_text(bw, 7, WL),
            ocr_text(bwi, 7, WL),
        ]
        # Devolver el más largo (mejor señal)
        return max(txts, key=lambda s: len(s or "")) if txts else ""

    raw1 = ocr_upper(band1) or ""
    raw2 = ocr_upper(band2) or ""

    # 1) Primera línea: puede venir con salto: "RODRIGUEZ ...\nLUIS"
    l1_main = raw1
    l1_extra = ""
    if "\n" in raw1:
        parts = [p.strip() for p in raw1.split("\n") if p.strip()]
        if parts:
            l1_main = parts[0]
            if len(parts) > 1:
                l1_extra = parts[-1]  # lo de abajo en L1

    first = clean_owner_line(l1_main.upper())
    picked_from = "strict"

    # 2) Decidir segunda línea (preferencia: extra de L1; si no, L2)
    second_raw = ""
    if l1_extra:
        second_raw = l1_extra
        picked_from = "from_l1_break"
    elif raw2:
        second_raw = raw2

    def accept_second(s: str) -> str:
        s = (s or "").upper().strip()
        if not s: return ""
        # parar en números o corchetes/dos puntos
        s = re.split(r"[\[\]:0-9]", s)[0].strip()
        if not s: return ""
        s = re.sub(r"[^\wÁÉÍÓÚÜÑ' -]+","", s).strip()
        # tokens muy cortos o en lista basura fuera
        if s in JUNK_2NDLINE: return ""
        if len(s) > 26: s = s[:26].rstrip()
        toks = [t for t in re.split(r"[^\wÁÉÍÓÚÜÑ'-]+", s) if t]
        if not toks: return ""
        # si es un único token, mejor que sea pista conocida o >=3 letras
        if len(toks) == 1 and (toks[0] not in NAME_HINTS and len(toks[0]) < 3):
            return ""
        return " ".join(toks)

    second = accept_second(second_raw)

    owner = first
    if second:
        # Evitar duplicados (p.ej. ya incluido)
        if second not in first:
            owner = f"{first} {second}"
            owner = owner.strip()

    dbg = {
        "band": [x0, y1a, x1, y2b],
        "y_line1": [y1a, y1b],
        "y_line2_hint": [y2a, y2b],
        "x0": x0, "x1": x1,
        "t1_raw": raw1,
        "t1_extra_raw": l1_extra,
        "t2_raw": raw2,
        "picked_from": picked_from,
    }
    return owner, dbg

# ──────────────────────────────────────────────────────────────────────────────
# Pipeline por filas
# ──────────────────────────────────────────────────────────────────────────────
def detect_rows_and_extract(bgr: np.ndarray,
                            annotate: bool = False,
                            annotate_names: bool = False) -> Tuple[Dict[str,str], dict, np.ndarray]:
    vis = bgr.copy()
    h, w = bgr.shape[:2]

    # zona de mini-mapas
    top = int(h * 0.12); bottom = int(h * 0.92)
    left = int(w * 0.06); right = int(w * 0.40)
    crop = bgr[top:bottom, left:right]
    mg, mp = color_masks(crop)

    mains  = contours_centroids(mg, min_area=(300 if FAST_MODE else 220))
    neighs = contours_centroids(mp, min_area=(220 if FAST_MODE else 160))
    if not mains:
        sides8 = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
        return sides8, {"rows": [], "raster":{"dpi":get_dpi()}}, vis

    mains_abs  = [(cx+left, cy+top, a) for (cx,cy,a) in mains]
    mains_abs.sort(key=lambda t: t[1])
    neighs_abs = [(cx+left, cy+top, a) for (cx,cy,a) in neighs]

    cols = find_columns_once(bgr)

    rows_dbg = []
    linderos = {k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]}
    used = set()

    for (mcx, mcy, _a) in mains_abs[:6]:
        best = None; best_d = 1e9
        for (nx, ny, _na) in neighs_abs:
            d = (nx-mcx)**2 + (ny-mcy)**2
            if d < best_d:
                best_d = d; best = (nx, ny)
        side = ""
        if best is not None and best_d < (w*0.30)**2:
            side = side8_of((mcx, mcy), best)

        owner, odbg = extract_owner_from_row(bgr, row_y=mcy, cols=cols)

        if side and owner and side not in used:
            linderos[side] = owner
            used.add(side)

        if annotate:
            cv2.circle(vis, (mcx, mcy), 10, (0,255,0), -1)
            if best is not None:
                cv2.circle(vis, best, 8, (0,0,255), -1)
                lbl_map = {
                    "norte":"N","noreste":"NE","este":"E","sureste":"SE",
                    "sur":"S","suroeste":"SW","oeste":"W","noroeste":"NW"
                }
                lbl = lbl_map.get(side,"")
                if lbl:
                    cv2.putText(vis, lbl, (best[0]-8, best[1]-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        if annotate_names and owner:
            x_tx = int(w*0.42)
            cv2.putText(vis, owner[:28], (x_tx, mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 3, cv2.LINE_AA)
            cv2.putText(vis, owner[:28], (x_tx, mcy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)

        rows_dbg.append({
            "row_y": mcy,
            "main_center": [mcx, mcy],
            "neigh_center": list(best) if best is not None else None,
            "side": side,
            "owner": owner,
            "ocr": odbg,
            "header_left_abs": cols.get("header_left_abs"),
            "x_nif_abs": cols.get("x_nif_abs"),
        })

    dbg = {"rows": rows_dbg, "raster":{"dpi":get_dpi()}}
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
        "FAST_DPI": get_int_env("FAST_DPI", 340),
        "PDF_DPI": get_int_env("PDF_DPI", 420),
        "cv2_flags": {"OTSU": bool(THRESH_OTSU)},
    }

@app.get("/preview", dependencies=[Depends(check_token)])
def preview_get(
    pdf_url: AnyHttpUrl = Query(...),
    labels: int = Query(0, description="1=mostrar rumbos (N/NE/E/...)"),
    names:  int = Query(0, description="1=mostrar nombre abreviado")
):
    pdf_bytes = fetch_pdf_bytes(str(pdf_url))
    try:
        bgr = page2_bgr(pdf_bytes)
        _linderos, _dbg, vis = detect_rows_and_extract(
            bgr, annotate=bool(labels), annotate_names=bool(names)
        )
    except Exception as e:
        err = str(e)
        blank = np.zeros((260, 900, 3), np.uint8)
        cv2.putText(blank, f"ERR: {err[:80]}", (10,130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        ok, png = cv2.imencode(".png", blank)
        return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

    ok, png = cv2.imencode(".png", vis)
    if not ok:
        raise HTTPException(status_code=500, detail="No se pudo codificar la vista previa.")
    return StreamingResponse(io.BytesIO(png.tobytes()), media_type="image/png")

@app.post("/preview", dependencies=[Depends(check_token)])
def preview_post(data: ExtractIn = Body(...),
                 labels: int = Query(0),
                 names: int = Query(0)):
    return preview_get(pdf_url=data.pdf_url, labels=labels, names=names)

@app.post("/extract", response_model=ExtractOut, dependencies=[Depends(check_token)])
def extract(data: ExtractIn = Body(...), debug: bool = Query(False)) -> ExtractOut:
    pdf_bytes = fetch_pdf_bytes(str(data.pdf_url))

    if TEXT_ONLY:
        return ExtractOut(
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
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
            linderos={k:"" for k in ["norte","noreste","este","sureste","sur","suroeste","oeste","noroeste"]},
            owners_detected=[],
            note=f"Excepción visión/OCR: {e}",
            debug={"exception": str(e)} if debug else None
        )


