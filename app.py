from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

# =========================
# 載入資源（啟動時只做一次）
# =========================

INDEX_PATH = "steam_games.index"
META_PATH = "steam_games_meta.csv"
MODEL_NAME = "all-MiniLM-L6-v2"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
meta = pd.read_csv(META_PATH)

print("Loading embedding model...")
model = SentenceTransformer(MODEL_NAME)

# =========================
# FastAPI app
# =========================

app = FastAPI()

# ---------- CORS（關鍵） ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 本地測試用，之後可改成指定 domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Request schema
# =========================

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# =========================
# API endpoint
# =========================

@app.post("/search")
def search_games(req: QueryRequest):
    # 將 query 轉成向量
    q_emb = model.encode([req.query], normalize_embeddings=True)

    # FAISS 搜尋
    scores, indices = index.search(q_emb, req.top_k)

    # 組合結果
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "rank": i + 1,
            "name": meta.iloc[idx]["name"],
            "score": float(scores[0][i])
        })

    return {
        "query": req.query,
        "results": results
    }

# =========================
# Root（可選，測試用）
# =========================

@app.get("/")
def root():
    return {"status": "API is running"}
