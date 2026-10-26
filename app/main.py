"""
Entry point aplikasi FastAPI — IT Helpdesk Assistant.

Alur endpoint utama:
  1. Menerima pertanyaan/permintaan dari user
  2. Meneruskan ke router (router.py) untuk menentukan intent
  3. Router memanggil RAG (rag.py) atau tool (tools.py) sesuai intent
  4. Mengembalikan response terstruktur ke client
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv

# Muat .env (untuk run lokal) SEBELUM import modul yang membaca environment variables.
# Di Docker, variabel dikirim via `--env-file .env`, sehingga file .env tidak ikut di image.
load_dotenv()

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from app import rag  # noqa: E402
from app.llm import LLMError  # noqa: E402
from app.router import handle_query  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("helpdesk")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Bangun index FAISS sekali saat startup (bukan di setiap request)."""
    try:
        await asyncio.to_thread(rag.build_index)
    except Exception:
        # Jangan gagalkan startup; rag.semantic_search akan mencoba lazy-build lagi.
        logger.exception("Gagal membangun FAISS index saat startup, akan dicoba ulang saat request")
    yield


app = FastAPI(
    title="IT Helpdesk Assistant API",
    description="RAG + router/tool-call untuk pertanyaan People & IT Operations.",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, examples=["Berapa jatah cuti tahunan?"])


class AskResponse(BaseModel):
    status: str
    intent: str
    data: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError):
    """Request tidak valid (misal query kosong) tetap dibalas dengan format terstruktur."""
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "intent": "invalid_request",
            "data": "Request tidak valid: field 'query' wajib diisi (1-1000 karakter).",
        },
    )


@app.get("/health")
async def health():
    """Endpoint sederhana untuk memastikan service berjalan."""
    return {"status": "ok", "index_ready": rag._index is not None}


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Endpoint utama. Menerima pertanyaan bebas dari user (bisa berupa
    pertanyaan informasi/kebijakan ATAU permintaan cek status tiket),
    lalu meneruskannya ke router untuk diproses.

    Penanganan error:
      - LLM gagal merespons (timeout, quota, key invalid) -> 503 terstruktur
      - Intent tidak dikenali -> 200 dengan status "error", intent "unknown" (dari router)
      - Error tak terduga -> 500 terstruktur (detail hanya di log server)
    """
    query = request.query.strip()
    if not query:
        return JSONResponse(
            status_code=422,
            content={"status": "error", "intent": "invalid_request", "data": "Query tidak boleh kosong."},
        )

    try:
        result = await handle_query(query)
        return AskResponse(**result)
    except LLMError as exc:
        logger.error("LLM error untuk query %r: %s", query, exc)
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "intent": "llm_unavailable",
                "data": "Maaf, layanan AI sedang tidak dapat merespons. Silakan coba beberapa saat lagi.",
            },
        )
    except Exception:
        logger.exception("Unexpected error untuk query %r", query)
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "intent": "internal_error",
                "data": "Terjadi kesalahan internal pada server. Silakan coba lagi.",
            },
        )
