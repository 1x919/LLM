import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

print("Loading data...")
df = pd.read_csv("data/steam_games_clean.csv")

texts = df["content"].tolist()

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding texts (this may take a few minutes)...")
embeddings = model.encode(
    texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
)

print("Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)

faiss.write_index(index, "steam_games.index")

df[["appid", "name", "content", "positive_ratio"]].to_csv(
    "steam_games_meta.csv", index=False
)

print("Index build complete with full metadata!")
