import requests
import time
import pandas as pd
import numpy as np
import re
import os

'''
Report Notes:
1. Tried with one query -> limited by 245 or less per subreddit (many were links/not enough selftext)
2. Tried with past year -> not enough data for r/Futurology
3. Switched from r/Technology to r/Futurology because r/Technology was mostly links
4. Switched to split-querry approach, for all time (subset later) to get around 1000+ posts before cleaning
'''

# max query to Reddit returns around 245 posts so running multiple queries per subreddit to get a larger sample size
# could get the same post appearing in different query categories, accounted for by removing duplicate post ids
SUBREDDITS = ["Futurology", "artificial", "singularity", "MachineLearning"]

QUERIES = [
    "artificial intelligence LLM language model", # relevant general terms
    "ChatGPT Claude Gemini Llama Mistral Grok DeepSeek", # specific models
    "machine learning deep learning neural network", # more relevant general terms
    "AGI generative AI foundation model", # more general models
    "OpenAI Anthropic Google DeepMind Meta AI", # AI companies
]

# across all time, later subset to post 2023 (tried year. but too small dataset)
SUBREDDIT_CONFIG = {
    "Futurology":      ("all",  200),
    "artificial":      ("all",  200),
    "singularity":     ("all",  200),
    "MachineLearning": ("all", 200),
}

MIN_WORD_COUNT  = 20            # minimum combined text word count so enough for feature vectors
# MAX_TEMPORAL_CONC = 0.50
POSTS_PER_SUB   = 250           # Goal for per query to get good spread of posts after cleaning
TEMPORAL_FLOOR  = "2023-01-01"  # AI Boom post-ChatGPT era

OUTPUT_RAW      = "reddit_ai_posts_RAW.csv"

# AI Keywords for confirmation and EDA
AI_KEYWORDS = [
    "ai", "artificial intelligence", "llm", "gpt", "chatgpt", "claude",
    "gemini", "llama", "mistral", "grok", "deepseek", "copilot",
    "machine learning", "deep learning", "neural network",
    "agi", "language model", "transformer", "openai", "anthropic", "google deepmind",
    "meta ai", "automation", "algorithm", "reinforcement", "diffusion",
    "generative", "multimodal", "embedding", "inference", "fine-tun",
    "pretrain", "foundation model", "ai model", "robotics", "computer vision",
]

# Reddit API unofficial access point
HEADERS = {"User-Agent": "pstat134-student-project/1.0 (educational use)"}


# Collect Data via Reddit's public search endpoint
def fetch_posts(subreddit: str, query: str, target: int = 200,
                time_filter: str = "year") -> pd.DataFrame:
    # Fetch posts from subreddit via Reddit's public search JSON endpoint.
    posts = []
    after = None
    page  = 0

    while len(posts) < target:
        url    = f"https://www.reddit.com/r/{subreddit}/search.json"
        params = {
            "q":           query,
            "restrict_sr": True,
            "sort":        "relevance", # sorted by relevance (keyword match frequency > upvote ratio > recency)
            "t":           time_filter,
            "limit":       100,        # Reddit max per request (go to next "page")
        }
        if after:
            params["after"] = after

        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=10)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"  [!] Request error on page {page}: {e}")
            break

        data     = r.json().get("data", {})
        children = data.get("children", []) # holds posts from subreddit

        if not children:
            break

        for child in children:
            p = child["data"]
            posts.append({
                "id":           p.get("id", ""),
                "title":        p.get("title", ""),
                "selftext":     p.get("selftext", ""),
                "upvote_ratio": p.get("upvote_ratio", None),
                "num_comments": p.get("num_comments", None),
                "score":        p.get("score", None),
                "is_self":      p.get("is_self", None),
                "edited":       p.get("edited", False),
                "subreddit":    p.get("subreddit", subreddit),
                "created_utc":  p.get("created_utc", None),
                "url":          p.get("url", ""),
                "permalink":    p.get("permalink", ""),
            })

        after = data.get("after")
        page += 1

        if not after:
            break

        time.sleep(1.2)   # under Reddit's rate limit

    return pd.DataFrame(posts)


def collect_all() -> pd.DataFrame:
    all_frames = []

    # for all subreddit -> for all queries
    for sub, (time_filter, target) in SUBREDDIT_CONFIG.items():
        sub_frames = []
        print(f"\n[{sub}] (t={time_filter}) running {len(QUERIES)} queries...")

        for i, query in enumerate(QUERIES):
            print(f"  Query {i+1}/{len(QUERIES)}: '{query[:55]}'")
            df_q = fetch_posts(sub, query, target=target, time_filter=time_filter) # get posts for query
            sub_frames.append(df_q)
            print(f"    → {len(df_q)} posts fetched")
            time.sleep(1.5)  # pause between queries for same subreddit

        df_sub = pd.concat(sub_frames, ignore_index=True).drop_duplicates(subset="id") # combind posts into one df
        print(f"r/{sub}: {len(df_sub)} unique posts from the queries")
        all_frames.append(df_sub)

    return pd.concat(all_frames, ignore_index=True)

if __name__ == "__main__":
    df_raw = collect_all()
    df_raw.to_csv(OUTPUT_RAW, index=False)
    print(f"\nsaved. (shape: {df_raw.shape})")

# Ran this a couple times with various min words, max posts per subreddit, with just one query, etc. 
# and landed on this variation to get a sufficient amount of data possible
