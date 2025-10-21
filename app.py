import os, glob, json
from typing import List, Dict
from fastapi import FastAPI, Body
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import requests

# ---------- Konfigurasi ----------
EMB_MODEL = os.getenv("EMB_MODEL", "intfloat/multilingual-e5-small")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")
TOP_K = int(os.getenv("TOP_K", "8"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))  # 0: purely relevance, 1: purely diversity

# ---------- Inisialisasi ----------
embedder = SentenceTransformer(EMB_MODEL)
client = chromadb.Client(Settings(anonymized_telemetry=False))
collection = client.get_or_create_collection(name="rag_demo", metadata={"hnsw:space":"cosine"})

app = FastAPI(title="RAG x SLM (PoC)")

# ---------- Util ----------
def e5_encode(texts: List[str]):
    # e5 butuh prefix "query: " / "passage: "
    return embedder.encode(texts, normalize_embeddings=True)

def embed_query(q: str):
    return e5_encode([f"query: {q}"])[0]

def embed_passages(texts: List[str]):
    return e5_encode([f"passage: {t}" for t in texts])

def mmr(doc_embeddings, query_embedding, top_k, lambda_mult=0.5):
    import numpy as np
    doc_embeddings = np.array(doc_embeddings)
    query_embedding = np.array(query_embedding)
    sim_to_query = doc_embeddings @ query_embedding
    selected, candidates = [], list(range(len(doc_embeddings)))
    if len(candidates) <= top_k: return candidates
    selected.append(max(candidates, key=lambda i: sim_to_query[i]))
    candidates.remove(selected[0])
    while len(selected) < top_k and candidates:
        def mmr_score(i):
            sim_div = max(doc_embeddings[i] @ doc_embeddings[j] for j in selected) if selected else 0
            return lambda_mult * sim_to_query[i] - (1 - lambda_mult) * sim_div
        next_i = max(candidates, key=mmr_score)
        selected.append(next_i)
        candidates.remove(next_i)
    return selected

def ollama_chat(model: str, system: str, user: str) -> str:
    url = "http://localhost:11434/api/chat"
    payload = {"model": model, "messages":[{"role":"system","content":system},{"role":"user","content":user}], "stream": False}
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    return r.json()["message"]["content"]

# ---------- Skema ----------
class DocIn(BaseModel):
    id: str
    text: str
    source: str = "unknown"
    page: int | None = None
    section: str | None = None

class AskIn(BaseModel):
    question: str
    top_k: int | None = None
    use_hybrid: bool = True
    locale: str = "id-ID"

# ---------- Ingest ----------
@app.post("/ingest")
def ingest(docs: List[DocIn]):
    texts = [d.text for d in docs]
    embs = embed_passages(texts)
    ids = [d.id for d in docs]
    metadatas = [d.model_dump() for d in docs]
    collection.add(ids=ids, documents=texts, embeddings=embs, metadatas=metadatas)
    return {"ok": True, "count": len(docs)}

# Ingest dari folder .txt sederhana
@app.post("/ingest_dir")
def ingest_dir(path: str = Body(..., embed=True)):
    files = glob.glob(os.path.join(path, "*.txt"))
    payload = []
    for fp in files:
        with open(fp, "r", encoding="utf-8", errors="ignore") as f:
            txt = f.read()
        # chunk sederhana
        CHUNK=1200; OVER=200
        i=0; n=0
        while i < len(txt):
            chunk = txt[i:i+CHUNK]
            payload.append(DocIn(id=f"{os.path.basename(fp)}::{n}", text=chunk, source=os.path.basename(fp)))
            n+=1; i+= (CHUNK-OVER)
    return ingest(payload)

# ---------- Retrieve + Generate ----------
@app.post("/ask")
def ask(q: AskIn):
    top_k = q.top_k or TOP_K

    # Dense search
    q_emb = embed_query(q.question)
    res = collection.query(query_embeddings=[q_emb], n_results=top_k*3, include=["documents","embeddings","metadatas"])
    docs = res["documents"][0]; metas = res["metadatas"][0]; embs = res["embeddings"][0]

    # (Opsional) Hybrid: gabung dengan BM25
    if q.use_hybrid:
        bm25 = BM25Okapi([d.split() for d in docs])
        scores = bm25.get_scores(q.question.split())
        # Fusion sederhana: z-score + penjumlahan
        import numpy as np
        cos_scores = (np.array([e @ q_emb for e in embs]))
        def z(x): return (x - x.mean()) / (x.std() + 1e-9)
        fused = z(cos_scores) + z(np.array(scores))
        # ambil kandidat lebih luas
        idx_sorted = np.argsort(-fused).tolist()
        docs = [docs[i] for i in idx_sorted]
        metas = [metas[i] for i in idx_sorted]
        embs  = [embs[i]  for i in idx_sorted]

    # MMR untuk diversity
    selected = mmr(embs, q_emb, top_k, lambda_mult=MMR_LAMBDA)
    ctx_blocks = []
    used_sources = []
    for i in selected:
        m = metas[i]; s = m.get("source","unknown")
        p = m.get("page"); tag = f"{s}{('#'+str(p)) if p is not None else ''}"
        used_sources.append(tag)
        ctx_blocks.append(f"[{tag}] {docs[i].strip()}")

    system = "Kamu asisten QA yang hanya menjawab dari konteks. Jika tidak ada info, katakan tidak tahu. Sertakan sumber."
    user = f"""[CONTEXT]
{chr(10).join(ctx_blocks)}

[QUESTION]
{q.question}

[FORMAT]
- Jawab ringkas (≤150 kata), bahasa {q.locale}
- Sertakan Sumber: [source#page, ...]
"""

    answer = ollama_chat(OLLAMA_MODEL, system, user)
    return {"answer": answer, "sources": list(dict.fromkeys(used_sources))}  # unique preserve order
