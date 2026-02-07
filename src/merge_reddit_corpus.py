import os
import re
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
INPUT_ROOT = "data/external/reddit_mental_health_dataset"
OUTPUT_FILE = "data/processed/reddit_unlabeled_merged.csv"

MIN_LEN = 20
MAX_SAMPLES = 50_000  # safe upper bound for Colab T4

TEXT_COLS = ["text", "statement", "body", "selftext", "content", "post"]
TITLE_COLS = ["title"]

# -----------------------------
# TEXT CLEANING
# -----------------------------
def normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# -----------------------------
# EXTRACT TEXT FROM ANY CSV
# -----------------------------
def extract_text(df: pd.DataFrame) -> pd.Series:
    title_col = next((c for c in TITLE_COLS if c in df.columns), None)
    body_col = next((c for c in TEXT_COLS if c in df.columns), None)

    if title_col and body_col and title_col != body_col:
        text = df[title_col].fillna("") + " " + df[body_col].fillna("")
    elif body_col:
        text = df[body_col]
    else:
        raise ValueError(f"No usable text column in {df.columns}")

    return text.apply(normalize)


# -----------------------------
# MAIN
# -----------------------------
def main():
    all_texts = []

    for root, _, files in os.walk(INPUT_ROOT):
        for file in files:
            if not file.endswith(".csv"):
                continue

            path = os.path.join(root, file)
            print(f"Reading: {path}")

            try:
                df = pd.read_csv(path)
                texts = extract_text(df)
                texts = texts[texts.str.len() >= MIN_LEN]
                all_texts.extend(texts.tolist())
            except Exception as e:
                print(f"Skipped {file}: {e}")

    merged = pd.DataFrame({"text": all_texts})
    merged = merged.drop_duplicates()

    # Cap size for Colab
    if len(merged) > MAX_SAMPLES:
        merged = merged.sample(n=MAX_SAMPLES, random_state=42)

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print("\nMerge complete")
    print("Total samples:", len(merged))
    print("Saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
