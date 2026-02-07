import re
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple

# ---------------------------
# CONFIG
# ---------------------------
RAW_DATA_DIR = "data/raw"
PROCESSED_DATA_DIR = "data/processed"

LABELED_FILE = f"{RAW_DATA_DIR}/sentiment_baseline.csv"
REDDIT_FILE = f"{RAW_DATA_DIR}/live_reddit_post.csv"

TRAIN_OUT = f"{PROCESSED_DATA_DIR}/train_5class.csv"
VAL_OUT = f"{PROCESSED_DATA_DIR}/val_5class.csv"
REDDIT_OUT = f"{PROCESSED_DATA_DIR}/reddit_unlabeled_clean.csv"

LABELS = ["normal", "stress", "anxiety", "depression", "suicidal"]

# ---------------------------
# TEXT NORMALIZATION
# ---------------------------
def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Expand common contractions (minimal, safe)
    contractions = {
        "idk": "i do not know",
        "im": "i am",
        "can't": "cannot",
        "won't": "will not",
        "dont": "do not",
        "doesnt": "does not",
        "didnt": "did not",
        "i'm": "i am",
        "it's": "it is",
        "that's": "that is",
        "there's": "there is"
    }

    for k, v in contractions.items():
        text = re.sub(rf"\b{k}\b", v, text)

    # Normalize elongated words: soooo → soo
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------
# LOAD & CLEAN LABELED DATA
# ---------------------------
def load_labeled_data() -> pd.DataFrame:
    df = pd.read_csv(LABELED_FILE)

    # Drop useless index column if present
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Explicit column mapping (dataset-specific, stable)
    df = df.rename(
        columns={
            "statement": "text",
            "status": "label"
        }
    )

    # Normalize labels
    df["label"] = df["label"].str.lower().str.strip()

    # Keep only the 5 production classes
    df = df[df["label"].isin(LABELS)]

    # Normalize text
    df["text"] = df["text"].apply(normalize_text)

    # Drop very short / empty rows
    df = df[df["text"].str.len() > 5]

    return df



# ---------------------------
# TRAIN / VAL SPLIT
# ---------------------------
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        stratify=df["label"]
    )
    return train_df, val_df


# ---------------------------
# LOAD & CLEAN REDDIT DATA
# ---------------------------
def load_unlabeled_reddit() -> pd.DataFrame:
    df = pd.read_csv(REDDIT_FILE)

    # Keep only what we need
    df = df[["title", "body"]]

    # Fill NaNs
    df["title"] = df["title"].fillna("")
    df["body"] = df["body"].fillna("")

    # Combine title + body
    df["text"] = (df["title"] + " " + df["body"]).str.strip()

    # Normalize text
    df["text"] = df["text"].apply(normalize_text)

    # Drop empty / very short posts
    df = df[df["text"].str.len() > 20]

    # Remove duplicates
    df = df.drop_duplicates(subset=["text"])

    return df[["text"]]



# ---------------------------
# MAIN PIPELINE
# ---------------------------
def main():
    print("🔹 Loading labeled data...")
    labeled_df = load_labeled_data()
    print(f"✔ Labeled samples: {len(labeled_df)}")

    print("🔹 Splitting train/validation...")
    train_df, val_df = split_data(labeled_df)

    print("🔹 Loading Reddit unlabeled data...")
    reddit_df = load_unlabeled_reddit()
    print(f"✔ Reddit samples: {len(reddit_df)}")

    print("🔹 Saving processed files...")
    train_df.to_csv(TRAIN_OUT, index=False)
    val_df.to_csv(VAL_OUT, index=False)
    reddit_df.to_csv(REDDIT_OUT, index=False)

    print("✅ Data preprocessing completed successfully")


if __name__ == "__main__":
    main()
