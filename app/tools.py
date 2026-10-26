"""
Modul Tools — fungsi-fungsi "aksi" yang dipanggil router ketika intent
BUKAN pertanyaan informasi/kebijakan, melainkan permintaan spesifik
yang butuh pengecekan data langsung (bukan retrieval semantik).

Pola ini mengikuti konsep Tool Integration di Sesi 12.3.
"""

import logging
from functools import lru_cache
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

MOCK_TICKETS_PATH = "data/mock_tickets.csv"


@lru_cache(maxsize=1)
def _load_tickets() -> pd.DataFrame:
    """Memuat mock_tickets.csv sekali saja (cache in-memory)."""
    df = pd.read_csv(MOCK_TICKETS_PATH, dtype=str).dropna(subset=["ticket_id"])
    df["ticket_id"] = df["ticket_id"].str.strip().str.upper()
    return df


def get_ticket_status(ticket_id: Optional[str]) -> str:
    """
    Mencari status tiket berdasarkan ticket_id pada data mock_tickets.csv.

    Pencarian bersifat deterministik (exact match), bukan semantic search,
    karena data status tiket adalah data terstruktur yang harus akurat.
    Fungsi ini tidak pernah melempar error ke pemanggil: seluruh kondisi
    gagal dikembalikan sebagai pesan yang informatif.
    """
    if not ticket_id or not str(ticket_id).strip():
        return (
            "Nomor tiket tidak ditemukan pada pertanyaan Anda. "
            "Mohon sertakan nomor tiket dengan format TICKET-XXXX, "
            "contoh: 'Apa status tiket TICKET-0005?'"
        )

    normalized_id = str(ticket_id).strip().upper()

    try:
        df = _load_tickets()
    except Exception:  # file hilang / rusak -> jangan crash-kan aplikasi
        logger.exception("Gagal memuat data tiket dari %s", MOCK_TICKETS_PATH)
        return "Maaf, data tiket sedang tidak dapat diakses. Silakan coba beberapa saat lagi."

    match = df[df["ticket_id"] == normalized_id]
    if match.empty:
        return (
            f"Tiket {normalized_id} tidak ditemukan di sistem. "
            "Mohon periksa kembali nomor tiket Anda, atau hubungi tim IT Support "
            "jika Anda yakin nomor tersebut benar."
        )

    row = match.iloc[0]
    return (
        f"Tiket {row['ticket_id']} atas nama {row['employee_name']} "
        f"(kategori: {row['category']}) dengan subjek \"{row['subject']}\" "
        f"saat ini berstatus: {row['status']}. "
        f"Pembaruan terakhir pada {row['last_update']}."
    )
