"""
Cleaning Pipeline
Subreddits: r/Futurology, r/artificial, r/singularity, r/MachineLearning
Topic: AI / Machine Learning posts only
"""

import requests
import time
import pandas as pd
import numpy as np
import re
import os

# Global Vars
SUBREDDITS = ["Futurology", "artificial", "singularity", "MachineLearning"] # list of subreddits

INPUT_RAW      = "reddit_ai_posts_RAW.csv"
OUTPUT_CLEAN    = "reddit_ai_posts_CLEAN.csv"
OUTPUT_BALANCED = "reddit_ai_posts_BALANCED.csv"

SUBREDDIT_CONFIG = {
    "Futurology":      ("all",  200),
    "artificial":      ("all",  200),
    "singularity":     ("all",  200),
    "MachineLearning": ("year", 200),
}

# AI Keyword Filter (Confirmation + EDA Opportunity)
AI_KEYWORDS = [
    "ai", "artificial intelligence", "llm", "gpt", "chatgpt", "claude",
    "gemini", "llama", "mistral", "grok", "deepseek", "copilot",
    "machine learning", "deep learning", "neural network",
    "agi", "language model", "transformer", "openai", "anthropic", "google deepmind",
    "meta ai", "automation", "algorithm", "reinforcement", "diffusion",
    "generative", "multimodal", "embedding", "inference", "fine-tun",
    "pretrain", "foundation model", "ai model", "robotics", "computer vision",
]

MIN_WORD_COUNT  = 20            # minimum combined text word count
MAX_TEMPORAL_CONC = 0.50        # warn if >50% posts in a single month
POSTS_PER_SUB   = 250           # target per subreddit after balancing
TEMPORAL_FLOOR  = "2023-01-01"  # post-ChatGPT era only

# Cleaning Functions

DELETED_MARKERS = {"[deleted]", "[removed]", ""}

def clean_text(text: str) -> str:
    """Strip URLs, markdown symbols, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\.\S+", "", text)      # remove URLs
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)          # remove markdown links
    text = re.sub(r"[*_~`>#]", "", text)               # remove markdown symbols
    text = re.sub(r"\n+", " ", text)                   # no newlines
    text = re.sub(r"\s+", " ", text).strip()           # remove extra whitespace
    return text

def word_count(text: str) -> int:
    return len(text.split()) if isinstance(text, str) else 0

def has_real_body(selftext: str) -> bool:
    """True if selftext is genuine content (not deleted/removed/empty)."""
    return isinstance(selftext, str) and selftext.strip() not in DELETED_MARKERS

def is_ai_related(title: str, body: str) -> bool:
    combined = (title + " " + body).lower()
    return any(kw in combined for kw in AI_KEYWORDS)

# Cleaning Data Pipeline
def clean_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    print(f"\n[Cleaning] Starting with {len(df)} raw posts...")

    # remove duplicate posts
    before = len(df)
    df = df.drop_duplicates(subset="id")
    print(f"  Dedup: removed {before - len(df)} duplicates -> {len(df)}")

    # remove link posts
    before = len(df)
    df = df[df["is_self"] == True]
    print(f"  Dropped link posts (is_self=False): removed {before - len(df)} → {len(df)}")

    # clean text
    df = df.copy()  # avoid SettingWithCopyWarning
    df["title"]    = df["title"].apply(clean_text)
    df["selftext"] = df["selftext"].apply(clean_text)
    print(f"  Text cleaned (URLs, markdown stripped)")

    # remove posts with missing or deleted body text
    df["has_body"] = df["selftext"].apply(has_real_body)
    before = len(df)
    df = df[df["has_body"] == True]
    print(f" Dropped deleted/missing body: removed {before - len(df)} -> {len(df)}")

    # combine text fields
    df["text"]            = (df["title"] + " " + df["selftext"]).str.strip()
    df["text_word_count"] = df["text"].apply(word_count)
    print(f"  Combined text fields")

    # threshold word count
    before = len(df)
    df = df[df["text_word_count"] >= MIN_WORD_COUNT]
    print(f"  6. Word count filter (>={MIN_WORD_COUNT}): removed {before - len(df)} -> {len(df)}")

    # AI keyword filter
    before = len(df)
    df = df[df.apply(lambda r: is_ai_related(r["title"], r["selftext"]), axis=1)]
    print(f"  AI keyword filter: removed {before - len(df)} -> {len(df)}")

    # 8. post 2023 posts AI/ChatGPT Boom to present day
    df["created_dt"] = pd.to_datetime(df["created_utc"], unit="s", errors="coerce")
    before = len(df)
    df = df[df["created_dt"] >= TEMPORAL_FLOOR]
    print(f"  8. Temporal floor (>= {TEMPORAL_FLOOR}): removed {before - len(df)} ->{len(df)}")

    # Normalize remaining fields
    df["edited"]   = df["edited"].apply(lambda x: False if x is False else True)

    # Final DF
    df = df[[
        "id", "subreddit", "title", "selftext", "text",
        "text_word_count", "has_body",
        "upvote_ratio", "num_comments", "score",
        "is_self", "edited", "created_utc", "created_dt",
        "url", "permalink"
    ]].reset_index(drop=True)

    return df

# Summary Function

def summarize(df: pd.DataFrame, label: str = ""):
    print(f"DATASET SUMMARY{' — ' + label if label else ''}")
    print(f"Total posts  : {len(df)}")
    print(f"Date range   : {df['created_dt'].min().date()} → {df['created_dt'].max().date()}")
    print()

    for sub, g in df.groupby("subreddit"):
        avg_words    = g["text_word_count"].mean()
        med_comments = g["num_comments"].median()
        print(f"  r/{sub:<22} {len(g):>4} posts | "
              f"avg words: {avg_words:>5.0f} | "
              f"median comments: {med_comments:.0f}")

    counts = df.groupby("subreddit").size()

    if counts.min() < 250:
        print("At least one subreddit has less than 250 posts.")
    else:
        print(f" All subreddits have at least 250 posts.")


if __name__ == "__main__":
    df_raw = pd.read_csv(INPUT_RAW)
    df_raw["created_utc"] = pd.to_numeric(df_raw["created_utc"], errors="coerce")

    # Clean
    df_clean = clean_pipeline(df_raw)

    # Type Casting
    df_clean["upvote_ratio"] = pd.to_numeric(df_clean["upvote_ratio"], errors="coerce")
    df_clean["num_comments"] = pd.to_numeric(df_clean["num_comments"], errors="coerce")
    df_clean["score"]        = pd.to_numeric(df_clean["score"], errors="coerce")

    summarize(df_clean, label="CLEAN")
    df_clean.to_csv(OUTPUT_CLEAN, index=False)
    print(f"Clean data saved to {OUTPUT_CLEAN}  (shape: {df_clean.shape})")