"""Text extraction for PDFs and images.

PDFs are read with PyMuPDF (better word spacing than pypdf); permission-only
encrypted PDFs are unlocked with an empty password. Image files and
image-only PDF pages fall back to OCR — pytesseract if a Tesseract binary is
present, otherwise easyocr (pure-Python, no system binary required).
"""
import os

import fitz  # PyMuPDF

try:
    import pytesseract
    from PIL import Image
    _PIL_OK = True
except Exception:
    _PIL_OK = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

_EASYOCR_READER = None


def _tesseract_available() -> bool:
    if not _PIL_OK:
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


def _get_easyocr():
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["fr", "en"], gpu=False)
    return _EASYOCR_READER


def ocr_available() -> bool:
    if _tesseract_available():
        return True
    try:
        import easyocr  # noqa: F401
        return True
    except Exception:
        return False


def _ocr_pil(img) -> str:
    if _tesseract_available():
        try:
            return pytesseract.image_to_string(img, lang="fra+eng")
        except Exception:
            try:
                return pytesseract.image_to_string(img)
            except Exception:
                pass
    try:
        import numpy as np
        reader = _get_easyocr()
        return "\n".join(reader.readtext(np.array(img), detail=0))
    except Exception:
        return ""


def extract_image(path: str) -> str:
    """OCR a standalone image file. Returns '' if no OCR backend works."""
    if not _PIL_OK:
        return ""
    try:
        return _ocr_pil(Image.open(path).convert("RGB")).strip()
    except Exception:
        return ""


def extract_pdf(path: str):
    """Yield (page_number, text) for each page. Unlocks permission-encrypted
    PDFs and OCRs image-only pages when an OCR backend is available."""
    pages = []
    ocr = ocr_available()
    with fitz.open(path) as doc:
        if doc.is_encrypted:
            doc.authenticate("")
        for i, page in enumerate(doc):
            try:
                text = page.get_text("text") or ""
            except Exception:
                text = ""
            if len(text.strip()) < 50 and ocr and _PIL_OK:
                try:
                    pix = page.get_pixmap(dpi=200)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    text = _ocr_pil(img)
                except Exception:
                    pass
            pages.append((i + 1, text))
    return pages


def extract_any(path: str):
    """Return a list of (page_number, text) for any supported file type."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_pdf(path)
    if ext in IMAGE_EXTS:
        return [(1, extract_image(path))]
    return []
