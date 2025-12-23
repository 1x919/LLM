import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

index = faiss.read_index("steam_games.index")
meta = pd.read_csv("steam_games_meta.csv")

model = SentenceTransformer("all-MiniLM-L6-v2")

def recommend(query, top_k=5):
    q_emb = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(q_emb, top_k)

    for i, idx in enumerate(indices[0]):
        print(f"{i+1}. {meta.iloc[idx]['name']}  (score={scores[0][i]:.3f})")

print("Query: open world survival crafting")
recommend("open world survival crafting")
