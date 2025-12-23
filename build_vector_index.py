import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import pickle
import os

# ---------- 設定 ----------
DATA_PATH = "data/steam_games_clean.csv"
MODEL_NAME = "all-MiniLM-L6-v2"   # 快、準、CPU 可跑
INDEX_PATH = "data/steam_faiss.index"
META_PATH = "data/steam_meta.pkl"
# --------------------------

print("讀取資料...")
df = pd.read_csv(DATA_PATH)

texts = df["content"].tolist()

print("載入 embedding 模型...")
model = SentenceTransformer(MODEL_NAME)

print("開始向量化（第一次會比較久）...")
embeddings = model.encode(
    texts,
    show_progress_bar=True,
    batch_size=64,
    normalize_embeddings=True
)

embeddings = np.array(embeddings).astype("float32")

print("建立 FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)  # cosine similarity
index.add(embeddings)

print("儲存 index 與 metadata...")
faiss.write_index(index, INDEX_PATH)

with open(META_PATH, "wb") as f:
    pickle.dump(df[["appid", "name", "content"]], f)

print("✅ 向量索引建立完成！")
print(f"總遊戲數：{len(df)}")
