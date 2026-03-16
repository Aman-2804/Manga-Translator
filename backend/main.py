"""FastAPI application for Manga Translator."""

import logging
import shutil
import uuid
from pathlib import Path
from threading import Thread
from typing import Dict, List

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pipeline.orchestrator import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Manga Translator", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("../uploads")
OUTPUT_DIR = Path("../outputs")

_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"}
_PDF_MIME = "application/pdf"


def _pdf_to_images(pdf_path: Path, out_dir: Path, start_idx: int = 1) -> int:
    """Convert every page of a PDF to a PNG in out_dir.

    Uses PyMuPDF (fitz) at 2× zoom (~144 DPI) which is sharp enough for
    OCR without being excessively large.  Returns the number of pages saved.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF uploads. Run: pip install pymupdf"
        ) from exc

    doc = fitz.open(str(pdf_path))
    mat = fitz.Matrix(2.0, 2.0)  # 2× zoom
    count = len(doc)
    for i in range(count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat)
        dest = out_dir / f"page_{start_idx + i:03d}.png"
        pix.save(str(dest))
    doc.close()
    logger.info("PDF → %d page image(s) in %s", count, out_dir)
    return count

# ── In-memory job status store ───────────────────────────────────────────────

job_status: Dict[str, dict] = {}


def _init_job(job_id: str, total_pages: int = 0) -> dict:
    entry = {
        "status": "queued",
        "progress": 0,
        "total_pages": total_pages,
        "completed_pages": 0,
        "step_label": "",
        "pages": [],
        "original_pages": [],
        "pdf": None,
        "failed": [],
        "error": None,
    }
    job_status[job_id] = entry
    return entry


def _run_translation_job(
    job_id: str,
    input_dir: str,
    output_dir: str,
    target_lang: str = "en",
    source_lang: str = "ja",
):
    """Background worker that runs the full pipeline and updates job_status."""
    entry = job_status[job_id]
    entry["status"] = "processing"

    def on_page_done(idx: int, total: int, page_name: str):
        entry["completed_pages"] = idx
        entry["total_pages"] = total
        logger.info("Job %s: page %d/%d done (%s)", job_id[:8], idx, total, page_name)

    def on_progress(pct: int, label: str):
        entry["progress"] = pct
        entry["step_label"] = label

    try:
        result = run_pipeline(
            input_path=input_dir,
            output_dir=output_dir,
            pdf_filename="translated.pdf",
            on_page_done=on_page_done,
            on_progress=on_progress,
            target_lang=target_lang,
            source_lang=source_lang,
        )

        entry["status"] = "done"
        entry["progress"] = 100
        entry["pages"] = [
            f"/download/{job_id}/{Path(p).name}" for p in result["pages"]
        ]
        entry["original_pages"] = [
            f"/download/{job_id}/{Path(p).name}" for p in result.get("original_pages", [])
        ]
        entry["pdf"] = f"/download/{job_id}/translated.pdf"
        entry["failed"] = result["failed"]

    except Exception as exc:
        logger.exception("Job %s failed: %s", job_id[:8], exc)
        entry["status"] = "failed"
        entry["error"] = str(exc)

    finally:
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.is_dir():
            shutil.rmtree(upload_dir, ignore_errors=True)
            logger.info("Cleaned up uploads for job %s", job_id[:8])


# ── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Manga Translator API ready")


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Upload manga page images (or a single PDF) for translation.

    Accepts one or more image files (PNG/JPG/WEBP) or a single PDF.
    Returns a job_id to use with /translate and /status.
    """
    job_id = str(uuid.uuid4())
    job_dir = UPLOAD_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for f in files:
        ext = Path(f.filename).suffix.lower()
        content_type = f.content_type or ""

        looks_like_pdf = ext == ".pdf" or _PDF_MIME in content_type
        looks_like_image = ext in _IMAGE_EXTS

        # Save to a temp file so we can inspect / convert it
        tmp_path = job_dir / f"upload_tmp_{saved}"
        with open(tmp_path, "wb") as fp:
            shutil.copyfileobj(f.file, fp)

        if looks_like_pdf or (not looks_like_image):
            # Try to open as PDF (handles files with no extension / wrong MIME type)
            try:
                import fitz
                page_count = _pdf_to_images(tmp_path, job_dir, start_idx=saved + 1)
                saved += page_count
                tmp_path.unlink(missing_ok=True)
                continue
            except Exception as pdf_exc:
                logger.error("PDF read error for %s: %s", f.filename, pdf_exc, exc_info=True)
                if looks_like_pdf:
                    # Was explicitly a PDF but failed — surface the error
                    tmp_path.unlink(missing_ok=True)
                    raise HTTPException(status_code=400, detail=f"Could not read PDF: {f.filename} ({pdf_exc})")
                # Not a PDF — fall through to image handling below

        if looks_like_image:
            dest = job_dir / f"page_{saved + 1:03d}{ext}"
            tmp_path.rename(dest)
            saved += 1
        else:
            tmp_path.unlink(missing_ok=True)

    if saved == 0:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(
            status_code=400,
            detail="No valid files found. Accepted formats: PNG, JPG, JPEG, WEBP, BMP, TIFF, PDF.",
        )

    _init_job(job_id)

    # Build sorted list of uploaded page URLs for client-side preview
    page_urls = [
        f"/uploaded/{job_id}/{p.name}"
        for p in sorted(job_dir.iterdir(), key=lambda x: x.name)
        if p.suffix.lower() in _IMAGE_EXTS
    ]

    logger.info("Upload complete: job=%s  files=%d", job_id[:8], saved)
    return JSONResponse(content={
        "job_id": job_id,
        "files_uploaded": saved,
        "page_urls": page_urls,
        "message": "Upload successful",
    })


@app.get("/uploaded/{job_id}/{filename}")
async def serve_uploaded(job_id: str, filename: str):
    """Serve an uploaded (or PDF-converted) page image."""
    file_path = UPLOAD_DIR / job_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(str(file_path))


@app.post("/translate/{job_id}")
async def translate(
    job_id: str,
    background_tasks: BackgroundTasks,
    target_lang: str = "en",
    source_lang: str = "ja",
):
    """Start translation for a previously uploaded job.

    Runs the pipeline in a background thread so the request returns
    immediately.  Poll GET /status/{job_id} for progress.
    """
    upload_dir = UPLOAD_DIR / job_id
    if not upload_dir.is_dir():
        raise HTTPException(status_code=404, detail="Job not found. Upload files first.")

    if job_id in job_status and job_status[job_id]["status"] == "processing":
        raise HTTPException(status_code=409, detail="Job is already being processed.")

    output_dir = OUTPUT_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    _init_job(job_id)
    job_status[job_id]["status"] = "processing"

    thread = Thread(
        target=_run_translation_job,
        args=(job_id, str(upload_dir), str(output_dir), target_lang, source_lang),
        daemon=True,
    )
    thread.start()

    logger.info("Translation started: job=%s", job_id[:8])
    return JSONResponse(content={
        "job_id": job_id,
        "status": "processing",
    })


@app.get("/status/{job_id}")
async def status(job_id: str):
    """Get the current status of a translation job."""
    if job_id not in job_status:
        upload_dir = UPLOAD_DIR / job_id
        if upload_dir.is_dir():
            return JSONResponse(content={
                "job_id": job_id,
                "status": "uploaded",
                "message": "Files uploaded. POST /translate/{job_id} to start.",
            })
        raise HTTPException(status_code=404, detail="Job not found.")

    entry = job_status[job_id]
    return JSONResponse(content={
        "job_id": job_id,
        "status": entry["status"],
        "progress": entry["progress"],
        "total_pages": entry["total_pages"],
        "completed_pages": entry["completed_pages"],
        "step_label": entry.get("step_label", ""),
        "pages": entry["pages"],
        "original_pages": entry.get("original_pages", []),
        "pdf": entry["pdf"],
        "failed": entry["failed"],
        "error": entry["error"],
    })


@app.get("/download/{job_id}/{filename}")
async def download(job_id: str, filename: str):
    """Download a translated page image or the final PDF."""
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    if file_path.suffix == ".pdf":
        return FileResponse(
            str(file_path),
            media_type="application/pdf",
            filename=f"translated_{job_id[:8]}.pdf",
        )

    return FileResponse(str(file_path), media_type="image/png")


@app.get("/health")
async def health():
    """Health check."""
    return {"status": "ok"}


app.mount("/", StaticFiles(directory="../frontend", html=True), name="frontend")
