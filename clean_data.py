import pandas as pd

# =========================
# 檔案路徑設定
# =========================
GAME_CSV = "data/steam.csv"
DESC_CSV = "data/steam_description_data.csv"
OUTPUT_CSV = "data/steam_games_clean.csv"

print("📥 讀取原始資料...")
games_df = pd.read_csv(GAME_CSV)
desc_df = pd.read_csv(DESC_CSV)

# =========================
# Owners 篩選（排除冷門）
# =========================
print("🔍 依 owners 篩選...")
if "owners" not in games_df.columns:
    raise ValueError("steam.csv 中找不到 owners 欄位")

remove_owners = ["0-20000"]
before_cnt = len(games_df)
games_df = games_df[~games_df["owners"].isin(remove_owners)]
print(f"Owners 篩選完成：{before_cnt} → {len(games_df)}")

# =========================
# 自動偵測 ID 欄位
# =========================
id_candidates = ["appid", "steam_appid", "app_id", "id", "game_id"]

def find_id_col(df):
    for col in id_candidates:
        if col in df.columns:
            return col
    return None

game_id_col = find_id_col(games_df)
desc_id_col = find_id_col(desc_df)

if game_id_col is None or desc_id_col is None:
    raise ValueError("找不到可用的 ID 欄位")

print(f"✔ 遊戲 ID 欄位：{game_id_col}")
print(f"✔ 描述 ID 欄位：{desc_id_col}")

# =========================
# 自動偵測描述欄位
# =========================
desc_candidates = [
    "description",
    "short_description",
    "about_the_game",
    "detailed_description",
]

desc_col = None
for col in desc_candidates:
    if col in desc_df.columns:
        desc_col = col
        break

if desc_col is None:
    raise ValueError("找不到遊戲描述欄位")

print(f"✔ 使用描述欄位：{desc_col}")

# =========================
# 合併資料
# =========================
desc_df = desc_df[[desc_id_col, desc_col]]
desc_df = desc_df.rename(columns={desc_id_col: game_id_col})

df = games_df.merge(desc_df, on=game_id_col, how="inner")
print(f"📊 合併後資料筆數：{len(df)}")

# =========================
# 正評比例計算與篩選
# =========================
if "positive_ratings" not in df.columns or "negative_ratings" not in df.columns:
    raise ValueError("缺少 positive_ratings / negative_ratings")

df["positive_ratio"] = df["positive_ratings"] / (
    df["positive_ratings"] + df["negative_ratings"]
)

before_cnt = len(df)
df = df[df["positive_ratio"] >= 0.5]
print(f"👍 正評篩選完成：{before_cnt} → {len(df)}")

# =========================
# 文字欄位清理
# =========================
text_cols = [c for c in ["genres", "steamspy_tags", "categories"] if c in df.columns]

def clean_text(text):
    if isinstance(text, str):
        return text.replace(";", ", ")
    return ""

for col in text_cols:
    df[col] = df[col].apply(clean_text)

df = df.dropna(subset=[desc_col])

# =========================
# ⭐ 核心：語意內容設計（關鍵）
# =========================
def build_content(row):
    parts = []

    # 標題
    parts.append(f"Game Title: {row['name']}")

    # 類型與玩法（最高權重）
    if "genres" in row and row["genres"]:
        parts.append(f"Primary Genres: {row['genres']}")

    if "steamspy_tags" in row and row["steamspy_tags"]:
        parts.append(f"Gameplay Tags: {row['steamspy_tags']}")

    if "categories" in row and row["categories"]:
        parts.append(f"Game Categories: {row['categories']}")

    # 描述（放最後，避免稀釋）
    parts.append(f"Game Description: {row[desc_col]}")

    return ". ".join(parts)

print("🧠 建立語意 content...")
df["content"] = df.apply(build_content, axis=1)

# =========================
# 輸出欄位整理
# =========================
df = df[[game_id_col, "name", "content", "positive_ratio"]]
df = df.rename(columns={game_id_col: "appid"})

df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
print(f"✅ 清理完成，輸出至 {OUTPUT_CSV}")
print(f"🎮 最終遊戲數量：{len(df)}")
