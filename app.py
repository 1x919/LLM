from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

# =========================
# 載入資源
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

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
# 工具：中文 → 英文（規則式）
# =========================
ZH_EN_MAP = {
    "生存": "survival",
    "開放世界": "open world",
    "建造": "base building",
    "製作": "crafting",
    "單人": "single player",
    "多人": "multiplayer",
    "恐怖": "horror",
    "動作": "action",
    "角色扮演": "RPG",
    "策略": "strategy",
    "模擬": "simulation",
    "沙盒": "sandbox",
    "冒險": "adventure"
}

def zh_to_en(query: str) -> str:
    translated = query
    for zh, en in ZH_EN_MAP.items():
        translated = translated.replace(zh, en)
    return translated

# =========================
# 工具：reranking
# =========================
def rerank_results(indices, scores, query_en):
    results = []

    keywords = query_en.lower().split()

    for idx, score in zip(indices, scores):
        row = meta.iloc[idx]

        boost = 0.0

        # keyword 命中加權
        content = f"{row.get('name', '')}".lower()
        for kw in keywords:
            if kw in content:
                boost += 0.05

        # 品質加權（如果有）
        quality = 1.0
        if "positive_ratio" in row:
            quality = 0.7 + row["positive_ratio"] * 0.3

        final_score = float(score) * quality + boost

        results.append({
            "name": row["name"],
            "score": final_score
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results

# =========================
# API endpoint
# =========================
@app.post("/search")
def search_games(req: QueryRequest):
    # 1️⃣ 中文 → 英文
    query_en = zh_to_en(req.query)

    # 2️⃣ embedding
    q_emb = model.encode([query_en], normalize_embeddings=True)

    # 3️⃣ FAISS 先抓多一點
    scores, indices = index.search(q_emb, 20)

    # 4️⃣ rerank
    reranked = rerank_results(indices[0], scores[0], query_en)

    # 5️⃣ 取前 top_k
    final = []
    for i, item in enumerate(reranked[:req.top_k]):
        final.append({
            "rank": i + 1,
            "name": item["name"],
            "score": round(item["score"], 3)
        })

    return {
        "query_original": req.query,
        "query_used": query_en,
        "results": final
    }

# =========================
# Root
# =========================
@app.get("/")
def root():
    return {"status": "AI search API running"}
