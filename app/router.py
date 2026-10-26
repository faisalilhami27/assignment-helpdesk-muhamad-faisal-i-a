"""
Router — menentukan apakah query user butuh RAG (informasi/kebijakan)
atau tool call (aksi spesifik, misal cek status tiket).

Pola ini mengikuti workflow router yang sudah dipelajari di Sesi 12.4:
  user input -> intent detection -> jika butuh data -> retrieval (RAG)
                                  -> jika butuh aksi -> tool call

Strategi intent detection (hybrid, 3 lapis):
  1. Rule-based fast path : query mengandung pola ticket_id (TICKET-XXXX)
                            -> pasti 'ticket_status' (deterministik, tanpa LLM).
  2. LLM classifier        : Gemini dengan structured output (enum) untuk
                            membedakan 'faq' / 'ticket_status' / 'unknown'.
  3. Keyword baseline      : dipakai sebagai fallback bila LLM gagal merespons,
                            sehingga layanan tetap berjalan (graceful degradation).
"""

import logging
import re
from typing import Literal, Optional

from google.genai import types
from pydantic import BaseModel

from app.llm import GENERATION_MODEL, get_client
from app.rag import answer_from_knowledge_base
from app.tools import get_ticket_status

logger = logging.getLogger(__name__)

VALID_INTENTS = ("faq", "ticket_status", "unknown")

# Menangkap variasi penulisan: "TICKET-0005", "ticket 5", "tiket #0005", "TICKET_12",
# "tiket no 3", "tiket nomor 0003"
TICKET_ID_PATTERN = re.compile(
    r"\b(?:ticket|tiket)\s*(?:no\.?|nomor|nomer|number|nr\.?)?\s*[-_#:]?\s*(\d{1,6})\b",
    re.IGNORECASE,
)

# Keyword baseline: kata yang mengindikasikan user ingin MENGECEK tiket miliknya.
_TICKET_KEYWORDS = (
    "status tiket", "status ticket", "cek tiket", "cek ticket", "tiket saya",
    "ticket saya", "progress tiket", "tiketku", "nomor tiket",
)
# Topik yang dicakup knowledge base People & IT Operations.
_FAQ_KEYWORDS = (
    "cuti", "izin", "sakit", "absen", "password", "vpn", "login", "email", "akses",
    "laptop", "perangkat", "software", "install", "resign", "onboarding", "karyawan baru",
    "gaji", "payroll", "rekening", "asuransi", "reimburse", "atasan", "hybrid", "wfh",
    "jam kerja", "lembur", "melahirkan", "paternity", "slack", "google meet",
    "workspace", "pelatihan", "kebijakan", "prosedur", "bagaimana cara",
)


class IntentClassification(BaseModel):
    intent: Literal["faq", "ticket_status", "unknown"]


_INTENT_PROMPT = """Anda adalah intent classifier untuk asisten internal People & IT Operations.
Klasifikasikan pertanyaan karyawan ke tepat SATU kategori:

- "faq": pertanyaan informasi/kebijakan/prosedur umum seputar HR & IT
  (cuti, izin, absensi, password, VPN, akses sistem, perangkat/laptop, software,
  onboarding, gaji/payroll, asuransi, reimbursement, kebijakan kerja, tools kolaborasi).
  Termasuk pertanyaan "bagaimana cara membuat/mengajukan tiket" (itu prosedur umum).
- "ticket_status": user ingin MENGECEK status/progres tiket helpdesk TERTENTU miliknya.
- "unknown": di luar cakupan HR & IT internal perusahaan (misal cuaca, resep, politik,
  matematika, obrolan umum) atau terlalu ambigu untuk dipahami.

Contoh:
- "Berapa jatah cuti tahunan?" -> faq
- "Gimana cara reset password VPN?" -> faq
- "Tolong cek progres tiket TICKET-0003" -> ticket_status
- "Tiket saya sudah sampai mana ya?" -> ticket_status
- "Siapa juara Piala Dunia 2022?" -> unknown

Pertanyaan: {query}"""


def extract_ticket_id(query: str) -> Optional[str]:
    """
    Mengekstrak ticket_id dari teks query dan menormalisasinya ke format
    'TICKET-XXXX' (sesuai mock_tickets.csv). Mengembalikan None bila tidak ada.

    Contoh: "status tiket ticket-5?" -> "TICKET-0005"
    """
    match = TICKET_ID_PATTERN.search(query or "")
    if not match:
        return None
    return f"TICKET-{int(match.group(1)):04d}"


def _keyword_intent(query: str) -> str:
    """Baseline intent detection berbasis keyword (tanpa LLM)."""
    q = query.lower()
    if any(k in q for k in _TICKET_KEYWORDS):
        return "ticket_status"
    if any(k in q for k in _FAQ_KEYWORDS):
        return "faq"
    return "unknown"


async def _llm_intent(query: str) -> str:
    response = await get_client().aio.models.generate_content(
        model=GENERATION_MODEL,
        contents=_INTENT_PROMPT.format(query=query),
        config=types.GenerateContentConfig(
            temperature=0.0,
            response_mime_type="application/json",
            response_schema=IntentClassification,
        ),
    )
    parsed = response.parsed
    if isinstance(parsed, IntentClassification):
        return parsed.intent
    # Fallback parsing manual bila SDK tidak mengisi .parsed
    return IntentClassification.model_validate_json(response.text).intent


async def detect_intent(query: str) -> str:
    """
    Menentukan intent dari query user. Mengembalikan salah satu:
      - "faq"           -> pertanyaan informasi/kebijakan, arahkan ke RAG
      - "ticket_status" -> permintaan cek status tiket, arahkan ke tool call
      - "unknown"       -> di luar cakupan / ambigu
    """
    # Lapis 1: pola ticket_id eksplisit -> deterministik, hemat biaya & latensi.
    if extract_ticket_id(query):
        logger.info("Intent via rule (ticket_id pattern): ticket_status")
        return "ticket_status"

    # Lapis 2: LLM classifier dengan structured output.
    try:
        intent = await _llm_intent(query)
        if intent in VALID_INTENTS:
            logger.info("Intent via LLM: %s", intent)
            return intent
    except Exception:
        logger.exception("Intent detection via LLM gagal, fallback ke keyword baseline")

    # Lapis 3: keyword baseline (graceful degradation).
    intent = _keyword_intent(query)
    logger.info("Intent via keyword baseline: %s", intent)
    return intent


async def handle_query(query: str) -> dict:
    """
    Fungsi utama yang dipanggil oleh endpoint /ask.
    Menjalankan intent detection, lalu merutekan ke RAG atau tool call yang sesuai.
    """
    intent = await detect_intent(query)

    if intent == "ticket_status":
        ticket_id = extract_ticket_id(query)
        data = get_ticket_status(ticket_id)  # aman untuk ticket_id None / tidak ada
        return {"status": "success", "intent": "ticket_status", "data": data}

    elif intent == "faq":
        data = await answer_from_knowledge_base(query)
        return {"status": "success", "intent": "faq", "data": data}

    else:
        return {
            "status": "error",
            "intent": "unknown",
            "data": (
                "Maaf, saya tidak memahami permintaan Anda atau permintaan tersebut di luar "
                "cakupan layanan People & IT Operations. Saya dapat membantu menjawab "
                "pertanyaan kebijakan HR/IT (misal cuti, password, perangkat) atau mengecek "
                "status tiket (contoh: 'Apa status tiket TICKET-0005?')."
            ),
        }
