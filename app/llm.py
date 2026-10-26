"""
Utilitas bersama untuk client Gemini API.

API key dan nama model WAJIB dibaca dari environment variables (tidak hardcode).
Client dibuat secara lazy agar aplikasi tetap bisa di-import (misal untuk
/health) dan memberikan pesan error yang jelas bila GEMINI_API_KEY belum diisi.
"""

import os
from typing import Optional

from google import genai
from google.genai import types

GENERATION_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite-preview")
EMBEDDING_MODEL = os.environ.get("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")

_client: Optional[genai.Client] = None


class LLMError(Exception):
    """Dilempar ketika pemanggilan Gemini API gagal (timeout, quota, key salah, dll)."""


def get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise LLMError(
                "GEMINI_API_KEY belum di-set. Salin .env.example menjadi .env lalu isi API key Anda."
            )
        _client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=20000),
        )
    return _client
