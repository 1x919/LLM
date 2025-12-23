import faiss
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

INDEX_PATH = "data/steam_faiss.index"
META_PATH = "data/steam_meta.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"

print("載入 index 與資料...")
index = faiss.read_index(INDEX_PATH)

with open(META_PATH, "rb") as f:
    df = pickle.load(f)

model = SentenceTransformer(MODEL_NAME)

query = "I like difficult single-player games "

query_vec = model.encode([query], normalize_embeddings=True).astype("float32")

D, I = index.search(query_vec, 5)

print("\n🔍 搜尋結果：")
for idx in I[0]:
    print("-", df.iloc[idx]["name"])
