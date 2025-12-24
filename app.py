from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
import re

# =========================
# 1. 載入資源
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
# 2. 模擬行為追蹤 (Behavior Tracking)
# =========================
# 儲存用戶最近推薦過的遊戲，用來計算「重複性」並提供新鮮感
user_history = []


def update_history(game_name):
    if game_name not in user_history:
        user_history.append(game_name)
    if len(user_history) > 5:
        user_history.pop(0)


# =========================
# 3. FastAPI 設定
# =========================
app = FastAPI(title="EED Counselor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# 4. 資料模型
# =========================
class QueryRequest(BaseModel):
    query: str
    mood: str = "neutral"  # 使用者情緒
    available_time: int = 60  # 可用時間（分鐘）
    top_k: int = 5


# =========================
# 5. 工具函式與 LLM 提示詞設定
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
    "冒險": "adventure",
    "壓力": "stress",
    "疲勞": "tired",
    "放鬆": "relax",
}


def zh_to_en(text: str) -> str:
    translated = text
    for zh, en in ZH_EN_MAP.items():
        translated = translated.replace(zh, en)
    return translated


def generate_eed_rationale(game_name, mood, time):
    system_prompt = """
    你是一位專門解決遊戲倦怠（EED）的 AI 心理諮商師。
    你的任務是根據玩家的心情和時間推薦遊戲。
    
    規則：
    1. 語義要溫暖、具備同理心。
    2. 禁止使用任何引用標註，絕對不能出現 或 。
    3. 直接輸出推薦文字，不要有開場白。
    """
    if time <= 45:
        time_text = "這款遊戲非常適合您現在有限的碎片時間，讓娛樂變得輕鬆可得。"
    else:
        time_text = f"既然您有 {time} 分鐘，這款遊戲能提供更深度的沉浸感，幫助您暫時忘卻現實壓力。"

    if "壓力" in mood or "疲" in mood:
        mood_text = (
            f"考慮到您目前的 {mood} 狀態，這款遊戲的節奏適中，能有效緩解遊戲倦怠感。"
        )
    else:
        mood_text = "這款遊戲的風格與您現在的心情非常契合，希望能帶來純粹的愉悅感。"

    # 若未來接入 LLM API，請在系統提示詞 (System Prompt) 加入：
    # "你是一位遊戲諮商師。嚴禁在輸出中包含任何 或 標籤。"

    return f"{mood_text} {time_text}"


def rerank_results(indices, scores, query_en, mood):
    results = []
    keywords = query_en.lower().split()

    for idx, score in zip(indices, scores):
        row = meta.iloc[idx]
        game_name = row.get("name", "")
        boost = 0.0

        # 1. 關鍵字加權
        content = f"{game_name} {row.get('content', '')}".lower()
        for kw in keywords:
            if kw in content:
                boost += 0.05

        # 2. 品質加權 (Positive Ratio)
        quality = 1.0
        if "positive_ratio" in row:
            quality = 0.7 + (row["positive_ratio"] * 0.3)

        # 3. 重複性檢查：如果遊戲在歷史紀錄中，降低分數以鼓勵「新鮮感」
        if game_name in user_history:
            boost -= 0.15

        final_score = float(score) * quality + boost

        results.append(
            {"name": game_name, "content": row.get("content", ""), "score": final_score}
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# =========================
# 6. API 端點
# =========================


@app.post("/search")
def search_games(req: QueryRequest):
    time_pref = "short session" if req.available_time < 45 else "immersive session"
    query_en = zh_to_en(req.query)

    # --- 權重強化版 ---
    # 1. 將 Requirement 放在最前面
    # 2. 重複兩次 Specific Requirements 以人為增加其在向量空間中的權重
    enhanced_query = (
        f"CORE REQUIREMENT: {query_en}. "
        f"The game MUST BE {query_en}. "
        f"Context: User is {req.mood} and needs a {time_pref} ({req.available_time} min)."
    )

    # 2. 向量搜尋 (Embedding)
    q_emb = model.encode([enhanced_query], normalize_embeddings=True)

    # 3. FAISS 檢索 (初步篩選 20 筆)
    scores, indices = index.search(q_emb, 20)

    # 4. 重排序 (加入重複性與品質分析)
    reranked = rerank_results(indices[0], scores[0], query_en, req.mood)

    # 5. 格式化結果並生成推薦理由
    final = []
    for i, item in enumerate(reranked[: req.top_k]):
        # 更新行為追蹤歷史
        update_history(item["name"])

        final.append(
            {
                "rank": i + 1,
                "name": item["name"],
                "score": round(item["score"], 3),
                "rationale": generate_eed_rationale(
                    item["name"], req.mood, req.available_time
                ),
            }
        )

    return {
        "status": "success",
        "user_context": {"mood": req.mood, "time": req.available_time},
        "results": final,
    }


@app.get("/")
def root():
    return {"status": "EED Counselor AI Assistant is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
