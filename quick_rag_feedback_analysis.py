#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Quick RAG-style Customer Feedback Analyzer (Beginner Friendly)
- Loads a CSV of customer feedback
- Runs simple sentiment classification with VADER (rule-based)
- Extracts "top issues" via frequent n-grams in negative feedback
- Retrieves similar past complaints using TF-IDF cosine similarity
- Saves an output JSON with a schema inspired by the assessment

USAGE (Windows Anaconda Prompt):
  python quick_rag_feedback_analysis.py --input sample_feedback.csv

Optional arguments for your own CSV:
  --input PATH_TO_YOUR_FILE.csv
  --text_col TEXT_COLUMN_NAME (default: text)
  --id_col ID_COLUMN_NAME (default: id)
  --timestamp_col TIMESTAMP_COLUMN_NAME (default: timestamp)
  --topk_neighbors 3
  --top_issues 5
"""
import argparse
import os
import json
import hashlib
from datetime import datetime

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sentiment (VADER) - small and fast
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def ensure_vader_downloaded():
    # Attempt to download VADER if not already present
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        try:
            nltk.download("vader_lexicon", quiet=True)
        except Exception:
            print("[WARN] Could not download VADER lexicon automatically. If you see an error, run:")
            print("       >>> import nltk; nltk.download('vader_lexicon')")

def label_vader(score):
    if score >= 0.05:
        return "positive"
    elif score <= -0.05:
        return "negative"
    return "neutral"

def short_hash(text):
    h = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return f"iss-{h[:4]}"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="sample_feedback.csv")
    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--id_col", type=str, default="id")
    parser.add_argument("--timestamp_col", type=str, default="timestamp")
    parser.add_argument("--topk_neighbors", type=int, default=3)
    parser.add_argument("--top_issues", type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(f"[ERROR] Input file not found: {args.input}")

    df = pd.read_csv(args.input)
    for col in [args.text_col, args.id_col, args.timestamp_col]:
        if col not in df.columns:
            raise SystemExit(f"[ERROR] Column '{col}' not found in CSV. Columns are: {list(df.columns)}")

    # 1) Sentiment
    ensure_vader_downloaded()
    sia = SentimentIntensityAnalyzer()
    df["text_proc"] = df[args.text_col].astype(str).fillna("")
    df["sent_score"] = df["text_proc"].map(lambda t: sia.polarity_scores(t)["compound"])
    df["sentiment"] = df["sent_score"].map(label_vader)

    # 2) Sentiment distribution over the dataset time span
    def to_timestamp(x):
        try:
            return pd.to_datetime(x, utc=True)
        except Exception:
            return pd.NaT
    df["ts"] = df[args.timestamp_col].map(to_timestamp)
    ts_min = pd.to_datetime(df["ts"].min(), utc=True)
    ts_max = pd.to_datetime(df["ts"].max(), utc=True)
    n_total = len(df)
    pos = (df["sentiment"] == "positive").sum() / max(n_total, 1)
    neg = (df["sentiment"] == "negative").sum() / max(n_total, 1)
    neu = (df["sentiment"] == "neutral").sum() / max(n_total, 1)

    sentiment_distribution = [{
        "window_start": ts_min.isoformat() if pd.notna(ts_min) else None,
        "window_end": ts_max.isoformat() if pd.notna(ts_max) else None,
        "positive": round(pos, 4),
        "negative": round(neg, 4),
        "neutral": round(neu, 4),
        "n": int(n_total)
    }]

    # 3) Top issues (very simple): frequent n-grams in negative feedback
    neg_df = df[df["sentiment"] == "negative"]
    top_issues = []
    if len(neg_df) > 0:
        cv = CountVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        X = cv.fit_transform(neg_df["text_proc"])
        freqs = X.sum(axis=0).A1
        vocab = cv.get_feature_names_out()
        pairs = sorted(zip(vocab, freqs), key=lambda x: x[1], reverse=True)
        for label, count in pairs[: args.top_issues]:
            # choose two exemplar doc IDs where this label appears
            mask = neg_df["text_proc"].str.contains(rf"\b{label}\b", case=False, regex=True, na=False)
            exemplars = neg_df.loc[mask, args.id_col].astype(str).head(2).tolist()
            top_issues.append({
                "cluster_id": short_hash(label),
                "label": label,
                "count": int(count),
                "trend": "n/a",
                "exemplars": exemplars
            })

    # 4) Simple "retrieval" for evidence (TF-IDF cosine similarity)
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1,2))
    tfidf_matrix = tfidf.fit_transform(df["text_proc"])
    sim = cosine_similarity(tfidf_matrix)
    retrieved_examples = []
    for i, row in df.iterrows():
        # We'll attach neighbors for negative & neutral only (more actionable)
        if row["sentiment"] in ("negative", "neutral"):
            sims = sim[i].copy()
            sims[i] = -1.0  # exclude self
            nn_idx = sims.argsort()[::-1][: args.topk_neighbors]
            neighbors = []
            for j in nn_idx:
                neighbors.append({
                    "id": str(df.iloc[j][args.id_col]),
                    "score": round(float(sims[j]), 4)
                })
            retrieved_examples.append({
                "query_doc": str(row[args.id_col]),
                "neighbors": neighbors
            })

    # 5) Suggest simple improvements based on top issues
    suggested_improvements = []
    for issue in top_issues:
        title = f"Address '{issue['label']}'"
        rationale = f"Frequent negative mentions ({issue['count']}) around '{issue['label']}'."
        evidence = []
        for ex in issue.get("exemplars", []):
            evidence.append({"id": ex, "excerpt": "", "similarity": 0.0})
        suggested_improvements.append({
            "title": title,
            "rationale": rationale,
            "evidence_examples": evidence
        })

    # Compose output JSON inspired by the assessment schema
    out = {
        "primary_id": "demo-run",
        "main_response": {
            "content": {
                "sentiment_distribution": sentiment_distribution,
                "top_issues": top_issues,
                "suggested_improvements": suggested_improvements,
                "retrieved_examples": retrieved_examples
            },
            "metadata": [{"key":"tenant","value":"demo"}],
            "confidence_score": 0.0,
            "uncertainty_factors": []
        },
        "supporting_data": [],
        "performance_metrics": {},
        "system_metadata": {
            "model_info": "vader+tfidf",
            "strategy_info": "mini-rag-v0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    }

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("[OK] Analysis complete.")
    print(f"     Saved: {out_path}")
    print("     Open it in any JSON viewer or VS Code to inspect the results.")
    print("     Tip: Re-run with your own CSV using --input path/to/your.csv and column flags if needed.")

if __name__ == "__main__":
    main()
