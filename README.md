# 🧠 RAG x SLM (Small Language Model) PoC

Proyek ini menunjukkan implementasi **Retrieval-Augmented Generation (RAG)** menggunakan **Small Language Model (SLM)** seperti `phi3:mini` dari Ollama.

## 🚀 Fitur
- Ingest dokumen `.txt` menjadi basis pengetahuan.
- Hybrid retrieval (BM25 + dense vector).
- MMR selection untuk diversity.
- FastAPI endpoint `/ask` untuk menjawab pertanyaan berbasis konteks.
- Dukungan bahasa Indonesia & sitasi sumber.

## 📦 Instalasi
1. Clone repositori ini atau ekstrak file ZIP.
2. Jalankan model Ollama terlebih dahulu:

```bash
ollama pull phi3:mini
ollama serve
```

3. Instal dependensi Python:

```bash
pip install -r requirements.txt
```

4. Jalankan API:

```bash
uvicorn app:app --reload --port 8000
```

## 📂 Struktur Folder
```
rag_slm_poc/
├── app.py
├── README.md
├── requirements.txt
└── docs_txt/   # folder opsional untuk file .txt yang akan di-ingest
```

## 🧩 Contoh Penggunaan
### Ingest dokumen
```bash
curl -X POST http://localhost:8000/ingest_dir -H "Content-Type: application/json" -d '{"path":"./docs_txt"}'
```

### Ajukan pertanyaan
```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"Apa tujuan proyek Prometix?"}'
```

## 🛠️ Catatan
- Untuk PDF, gunakan `pdfplumber` atau `pymupdf` untuk konversi ke `.txt` terlebih dahulu.
- Parameter bisa diatur melalui environment variable:
  - `EMB_MODEL` → model embedding (default: multilingual-e5-small)
  - `OLLAMA_MODEL` → model Ollama (default: phi3:mini)
  - `TOP_K`, `MMR_LAMBDA` untuk tuning hasil retrieval.

---
Dibuat oleh **Kiki Ginanjar** 🧑‍💻
