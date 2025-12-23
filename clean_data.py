import pandas as pd

# ===== 請你只需要改這兩行 =====
GAME_CSV = "data/steam.csv"
DESC_CSV = "data/steam_description_data.csv"
# =================================

games_df = pd.read_csv(GAME_CSV)
desc_df = pd.read_csv(DESC_CSV)

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

print("合併後資料筆數：", len(df))

# ---------- 組語意內容 ----------
text_cols = [col for col in ["genres", "steamspy_tags", "categories"] if col in df.columns]

df = df[[game_id_col, "name", desc_col] + text_cols]
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

df = df[[game_id_col, "name", "content"]]
df = df.rename(columns={game_id_col: "appid"})

df.to_csv("data/steam_games_clean.csv", index=False, encoding="utf-8")

print("清理完成，剩餘遊戲數量：", len(df))
