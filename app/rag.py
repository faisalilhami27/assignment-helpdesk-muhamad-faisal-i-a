"""
Modul RAG — mengubah knowledge base menjadi embedding, menyimpannya ke FAISS,
melakukan semantic search, lalu menghasilkan jawaban menggunakan LLM.

Alur ini serupa dengan yang sudah dibangun di Assignment 3 (Semantic Search &
LLM Application) dan Sesi 11 (Building a Simple RAG System) — di assignment ini,
alur tersebut dibungkus menjadi fungsi yang dipanggil oleh router.py.
"""

import logging
import os
import threading

import faiss
import numpy as np
import pandas as pd
from google.genai import types

from app.llm import EMBEDDING_MODEL, GENERATION_MODEL, LLMError, get_client

logger = logging.getLogger(__name__)

KNOWLEDGE_BASE_PATH = "data/it_helpdesk_knowledge_base.csv"

# Ambang minimal cosine similarity agar sebuah dokumen dianggap relevan.
# Di bawah ambang ini, sistem menyatakan tidak tahu (mengurangi halusinasi).
MIN_RELEVANCE_SCORE = float(os.environ.get("RAG_MIN_SCORE", "0.7"))
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "3"))
_EMBED_BATCH_SIZE = 100

NO_ANSWER_MESSAGE = (
    "Maaf, saya belum menemukan informasi terkait pertanyaan tersebut di knowledge base "
    "People & IT Operations. Silakan hubungi tim HR atau IT Support secara langsung, "
    "atau buat tiket IT agar dapat ditindaklanjuti."
)

# Index dibangun SEKALI saat startup (dipanggil dari lifespan FastAPI di main.py),
# lalu disimpan di variabel modul ini sebagai cache in-memory.
_index = None
_documents = None
_df = None
_index_lock = threading.Lock()


def _get_embedding(text: str, task_type: str = "RETRIEVAL_QUERY"):
    """Embedding satu teks (dipakai untuk query user)."""
    try:
        result = get_client().models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type=task_type),
        )
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Gagal membuat embedding: {exc}") from exc
    return result.embeddings[0].values


def _get_embeddings_batch(texts: list[str]) -> np.ndarray:
    """Embedding banyak dokumen sekaligus (batch) untuk efisiensi saat indexing."""
    vectors = []
    try:
        for start in range(0, len(texts), _EMBED_BATCH_SIZE):
            batch = texts[start : start + _EMBED_BATCH_SIZE]
            result = get_client().models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch,
                config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
            )
            vectors.extend(e.values for e in result.embeddings)
    except LLMError:
        raise
    except Exception as exc:
        raise LLMError(f"Gagal membuat embedding knowledge base: {exc}") from exc
    return np.array(vectors, dtype="float32")


def build_index():
    """
    Membangun index FAISS dari knowledge base CSV.

    Langkah: load data -> susun unit retrievable -> embedding -> simpan ke FAISS.
    Unit retrievable = satu baris FAQ (kategori + pertanyaan + jawaban), karena
    setiap FAQ sudah merupakan satu unit informasi yang utuh dan pendek
    (tidak perlu chunking tambahan).

    Vektor dinormalisasi (L2) lalu disimpan di IndexFlatIP, sehingga inner product
    setara dengan cosine similarity (skor 0..1 yang mudah diberi ambang).
    """
    global _index, _documents, _df

    with _index_lock:
        if _index is not None:
            return _index

        df = pd.read_csv(KNOWLEDGE_BASE_PATH)
        df = df.dropna(subset=["question", "answer"]).reset_index(drop=True)
        df["category"] = df["category"].fillna("Umum")

        documents = (
            "Kategori: " + df["category"].str.strip()
            + "\nPertanyaan: " + df["question"].str.strip()
            + "\nJawaban: " + df["answer"].str.strip()
        ).tolist()

        embeddings = _get_embeddings_batch(documents)
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)

        _df, _documents, _index = df, documents, index
        logger.info("FAISS index siap: %d dokumen, dimensi %d", index.ntotal, index.d)
        return _index


def semantic_search(query: str, top_k: int = DEFAULT_TOP_K):
    """
    Semantic search terhadap index FAISS.

    Mengembalikan list dict berisi faq_id, category, question, answer, dan score
    (cosine similarity), diurutkan dari yang paling relevan.
    """
    if _index is None:
        build_index()  # fallback lazy-loading bila startup indexing gagal

    query_vec = np.array([_get_embedding(query, "RETRIEVAL_QUERY")], dtype="float32")
    faiss.normalize_L2(query_vec)

    top_k = max(1, min(top_k, _index.ntotal))
    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = _df.iloc[idx]
        results.append(
            {
                "faq_id": int(row["faq_id"]),
                "category": row["category"],
                "question": row["question"],
                "answer": row["answer"],
                "score": float(score),
            }
        )
    return results


def _build_prompt(query: str, contexts: list[dict]) -> str:
    """Menyusun prompt terstruktur: Persona / Task / Context / Format."""
    context_block = "\n\n".join(
        f"[Dokumen {i}] (Kategori: {c['category']})\n"
        f"Pertanyaan: {c['question']}\nJawaban: {c['answer']}"
        for i, c in enumerate(contexts, start=1)
    )
    return f"""### PERSONA
Anda adalah "IT Helpdesk Assistant", asisten internal tim People & IT Operations
yang ramah, profesional, dan akurat dalam menjawab pertanyaan karyawan.

### TASK
Jawab pertanyaan karyawan HANYA berdasarkan informasi pada bagian CONTEXT.
- Jangan menambahkan informasi, angka, atau kebijakan yang tidak ada di CONTEXT.
- Jika CONTEXT tidak memuat jawaban yang relevan, katakan dengan jujur bahwa
  informasi tersebut belum tersedia dan sarankan menghubungi tim HR/IT Support.

### CONTEXT
{context_block}

### FORMAT
- Gunakan Bahasa Indonesia yang jelas dan sopan.
- Jawaban singkat dan langsung ke inti (maksimal 3-4 kalimat).
- Boleh gunakan poin bernomor bila berisi langkah-langkah.
- Jangan menyebut kata "Dokumen" atau "CONTEXT" di dalam jawaban.

### PERTANYAAN KARYAWAN
{query}

### JAWABAN
"""


async def answer_from_knowledge_base(query: str) -> str:
    """
    Fungsi yang dipanggil router.py ketika intent = 'faq'.
    Menjalankan semantic search, menyusun prompt dengan context hasil retrieval,
    lalu memanggil LLM untuk menghasilkan jawaban.
    """
    results = semantic_search(query)
    relevant = [r for r in results if r["score"] >= MIN_RELEVANCE_SCORE]

    logger.info(
        "Retrieval untuk %r: %s",
        query,
        [(r["faq_id"], round(r["score"], 3)) for r in results],
    )

    if not relevant:
        # Guardrail: tidak ada konteks yang cukup relevan -> jangan paksa LLM menjawab.
        return NO_ANSWER_MESSAGE

    prompt = _build_prompt(query, relevant)
    try:
        response = await get_client().aio.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
    except Exception as exc:
        raise LLMError(f"LLM gagal menghasilkan jawaban: {exc}") from exc

    answer = (response.text or "").strip()
    if not answer:
        raise LLMError("LLM mengembalikan jawaban kosong.")
    return answer
