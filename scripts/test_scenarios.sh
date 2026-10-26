#!/usr/bin/env bash
# Menguji endpoint /ask dengan 5 skenario wajib + beberapa edge case.
# Pemakaian: ./scripts/test_scenarios.sh [BASE_URL]   (default http://localhost:8000)
set -u
BASE_URL="${1:-http://localhost:8000}"

ask() {
  local label="$1" query="$2"
  echo "=== ${label}"
  echo "Query : ${query}"
  curl -s -w "\nHTTP  : %{http_code}\n" -X POST "${BASE_URL}/ask" \
    -H "Content-Type: application/json" \
    -d "{\"query\": \"${query}\"}"
  echo
}

echo "Health: $(curl -s "${BASE_URL}/health")"
echo

# --- 5 skenario wajib ---
ask "S1 [RAG] kebijakan cuti"         "Berapa jatah cuti tahunan untuk karyawan tetap?"
ask "S2 [RAG] reset password VPN"     "Bagaimana cara reset password VPN perusahaan?"
ask "S3 [TOOL] tiket valid"           "Tolong cek status tiket TICKET-0005 saya"
ask "S4 [TOOL] tiket tidak ada"       "Apa status tiket TICKET-0099?"
ask "S5 [OOS] di luar cakupan"        "Siapa juara Piala Dunia 2022?"

# --- edge case tambahan ---
ask "E1 [TOOL] ticket_status tanpa ID"     "Tiket saya sudah sampai mana ya prosesnya?"
ask "E2 [TOOL] format ID tidak baku"       "cek tiket no 3 dong"
ask "E3 [RAG] FAQ tapi tak ada di KB"      "Apakah kantor menyediakan parkir motor untuk karyawan?"
ask "E4 [RAG] prosedur membuat tiket"      "Bagaimana cara mengajukan permintaan laptop baru?"

echo "=== E5 [VALIDATION] query kosong"
curl -s -w "\nHTTP  : %{http_code}\n" -X POST "${BASE_URL}/ask" \
  -H "Content-Type: application/json" -d '{"query": ""}'
