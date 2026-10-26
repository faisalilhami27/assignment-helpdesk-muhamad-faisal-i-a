# ---- Base image ringan & stabil (faiss-cpu tersedia sebagai wheel) ----
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies dulu agar layer ini ter-cache selama requirements tidak berubah
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code & data (path relatif "data/..." mengacu ke WORKDIR /app)
COPY app/ ./app/
COPY data/ ./data/

# Jalankan sebagai non-root user
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

# API key TIDAK dimasukkan ke image — diberikan saat runtime:
#   docker run --env-file .env -p 8000:8000 it-helpdesk-assistant
# Host 0.0.0.0 wajib agar endpoint dapat diakses dari luar container.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
