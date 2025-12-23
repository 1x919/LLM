import pandas as pd

# ===== 請你只需要改這兩行 =====
GAME_CSV = "data/steam.csv"
DESC_CSV = "data/steam_description_data.csv"
# =================================

games_df = pd.read_csv(GAME_CSV)
desc_df = pd.read_csv(DESC_CSV)

# ---------- 先依 owners 篩選 ----------
if "owners" not in games_df.columns:
    raise ValueError("steam.csv 中找不到 owners 欄位")

# 移除 owners 為 0-20000 與 20000~50000
remove_owners = ["0-20000"]
before_cnt = len(games_df)

games_df = games_df[~games_df["owners"].isin(remove_owners)]

print(f"Owners 篩選完成：{before_cnt} → {len(games_df)}")

# ---------- 自動找 ID 欄位 ----------
id_candidates = ["appid", "steam_appid", "app_id", "id", "game_id"]

game_id_col = None
for col in id_candidates:
    if col in games_df.columns:
        game_id_col = col
        break

desc_id_col = None
for col in id_candidates:
    if col in desc_df.columns:
        desc_id_col = col
        break

if game_id_col is None or desc_id_col is None:
    raise ValueError("找不到可用的 ID 欄位")

print(f"Games ID 欄位：{game_id_col}")
print(f"Description ID 欄位：{desc_id_col}")

# ---------- 自動找描述欄位 ----------
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
    raise ValueError("找不到 description 欄位")

print(f"使用描述欄位：{desc_col}")

# ---------- 合併 ----------
desc_df = desc_df[[desc_id_col, desc_col]]
desc_df = desc_df.rename(columns={desc_id_col: game_id_col})

df = games_df.merge(desc_df, on=game_id_col, how="inner")

# ---------- 計算正評比例 ----------
if "positive_ratings" in df.columns and "negative_ratings" in df.columns:
    df["positive_ratio"] = df["positive_ratings"] / (
        df["positive_ratings"] + df["negative_ratings"]
    )
else:
    raise ValueError("找不到 positive_ratings 或 negative_ratings 欄位")

# ---------- 移除正評比例 < 50% ----------
before_cnt = len(df)
df = df[df["positive_ratio"] >= 0.5]
print(f"正評比例篩選完成：{before_cnt} → {len(df)}")

print("合併後資料筆數：", len(df))

# ---------- 組語意內容 ----------
text_cols = [col for col in ["genres", "steamspy_tags", "categories"] if col in df.columns]

df = df[[game_id_col, "name", desc_col, "positive_ratio"] + text_cols]

df = df.dropna(subset=[desc_col])

def clean_text(text):
    if isinstance(text, str):
        return text.replace(";", ", ")
    return ""

for col in text_cols:
    df[col] = df[col].apply(clean_text)

def build_content(row):
    parts = [
        f"Game: {row['name']}",
        f"Description: {row[desc_col]}",
    ]
    for col in text_cols:
        if row[col]:
            parts.append(f"{col.capitalize()}: {row[col]}")
    return ". ".join(parts)

df["content"] = df.apply(build_content, axis=1)

df = df[[game_id_col, "name", "content", "positive_ratio"]]

df = df.rename(columns={game_id_col: "appid"})

df.to_csv("data/steam_games_clean.csv", index=False, encoding="utf-8")

print("清理完成，剩餘遊戲數量：", len(df))
