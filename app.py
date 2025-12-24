from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import ollama
import json
import os
import re

# 強制離線模式（確保不會因為連不到 HuggingFace 而崩潰）
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

# =========================
# 1. 資源與模型設定
# =========================
INDEX_PATH = "steam_games.index"
META_PATH = "steam_games_meta.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"

print("Loading FAISS index...")
index = faiss.read_index(INDEX_PATH)

print("Loading metadata...")
meta = pd.read_csv(META_PATH)

print("Loading embedding model...")
# --- 修正處：直接初始化 model，不要重複傳遞物件 ---
model = SentenceTransformer(MODEL_NAME, local_files_only=True)

# =========================
# 2. 模擬行為追蹤
# =========================
user_history = []


def update_history(game_name):
    if game_name not in user_history:
        user_history.append(game_name)
    if len(user_history) > 5:
        user_history.pop(0)


# =========================
# 3. FastAPI 設定
# =========================
app = FastAPI(title="EED Counselor AI - Pro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    mood: str = "neutral"
    available_time: int = 60
    top_k: int = 3


# =========================
# 4. LLM 核心邏輯
# =========================


def expand_query_with_llm(user_query, mood):
    prompt = f"玩家目前心情：{mood}。搜尋需求：{user_query}。請將此需求轉化為一段 30 字以內的遊戲特徵描述，直接輸出描述文字。"
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        expanded = response["message"]["content"].strip()
        return f"{user_query} {expanded}"
    except:
        return user_query


def llm_rerank(candidates, mood, available_time):
    game_list_str = "\n".join(
        [f"- {c['name']}: {str(c['content'])[:150]}" for c in candidates]
    )

    prompt = f"""
    作為遊戲心理諮商師，請從以下候選清單中，挑選出最適合目前心情「{mood}」且可用時間為「{available_time} 分鐘」的 3 款遊戲。
    請考慮玩家的心理狀態，並以繁體中文的語境進行思考。
    
    候選清單：
    {game_list_str}

    請嚴格依照以下 JSON 格式回傳：
    [ {{"name": "遊戲名稱1"}}, {{"name": "遊戲名稱2"}}, {{"name": "遊戲名稱3"}} ]
    """
    try:
        # 增加 format='json' 確保 Ollama 輸出正確格式
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            format="json",
        )
        content = response["message"]["content"].strip()

        # 使用正則表達式提取 JSON 陣列，避免廢話干擾
        json_match = re.search(r"\[.*\]", content, re.DOTALL)
        if json_match:
            ranked_names = json.loads(json_match.group(0))
        else:
            ranked_names = json.loads(content)

        final_list = []
        for r in ranked_names:
            match = next((c for c in candidates if c["name"] == r.get("name")), None)
            if match:
                final_list.append(match)

        return final_list if final_list else candidates[:3]
    except Exception as e:
        print(f"Rerank Error: {e}")
        return candidates[:3]


def generate_eed_rationale(game_name, game_content, mood, time):
    """
    為選出的遊戲生成暖心的推薦理由
    """
    # 在 Prompt 中加入語系要求
    prompt = f"""
    你是一位專門解決「遊戲倦怠 (EED)」的 AI 心理諮商師。
    玩家目前的狀態：
    - 心情：{mood}
    - 可用時間：{time} 分鐘
    
    推薦遊戲資訊：
    - 名稱：{game_name}
    - 遊戲特點：{str(game_content)[:300]}

    任務：
    請寫一段 50 字以內、暖心且具備同理心的推薦文字，說明這款遊戲如何幫助玩家。
    
    規則：
    1. 務必使用「繁體中文」回答。
    2. 直接輸出推薦內容，不要有開場白。
    3. 語氣要溫暖，像是一位關心玩家的朋友或諮商師。
    """
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}]
        )
        return response["message"]["content"].strip()
    except:
        return "這款遊戲非常適合您現在的心情，希望能為您帶來純粹的愉悅感。"


# =========================
# 5. API 端點
# =========================


@app.post("/search")
def search_games(req: QueryRequest):
    # Step 1: LLM 查詢理解
    enhanced_query = expand_query_with_llm(req.query, req.mood)

    # Step 2: 向量搜尋 (初步篩選 15 筆)
    q_emb = model.encode([enhanced_query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, 15)

    # Step 3: 候選名單準備
    candidates = []
    for idx, score in zip(indices[0], scores[0]):
        row = meta.iloc[idx]
        candidates.append(
            {
                "appid": int(row.get("appid", 0)),
                "name": row.get("name", ""),
                "content": row.get("content", ""),
                "base_score": float(score),
            }
        )

    # Step 4: LLM 重排序
    final_top_3 = llm_rerank(candidates, req.mood, req.available_time)

    # Step 5: 生成最終結果
    results = []
    for i, item in enumerate(final_top_3[:3]):
        update_history(item["name"])
        rationale = generate_eed_rationale(
            item["name"], item["content"], req.mood, req.available_time
        )
        results.append(
            {
                "rank": i + 1,
                "appid": item["appid"],
                "name": item["name"],
                "score": round(item["base_score"], 3),
                "rationale": rationale,
            }
        )

    return {"status": "success", "results": results}


@app.get("/")
def root():
    return {"status": "EED Counselor AI Pro (Ollama) is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
