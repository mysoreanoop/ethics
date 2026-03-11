import json
import os
import argparse
import re
import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from bertopic import BERTopic
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
import torch
from transformers import pipeline


DEFAULT_DATA_GLOB = "data/*/posts_*.json"

ABSA_ROOT = Path("results/absa")
OUTPUT_ROOT = Path("results/nlu")

GOEMOTIONS_MODEL = "SamLowe/roberta-base-go_emotions"
TEXT_MAX_CHARS = 1200
BATCH_SIZE = 64
TARGET_NUM_TOPICS = 20

NOISE_TERMS = {
    "amp",
    "png",
    "jpg",
    "jpeg",
    "webp",
    "http",
    "https",
    "www",
    "com",
    "preview",
    "width",
    "format",
    "auto",
    "reddit",
    "redd",
}

SENTIMENT_TO_SCORE = {
    "positive": 1,
    "neutral": 0,
    "negative": -1,
}


@dataclass
class DocumentRow:
    source_file: str
    subreddit: str
    post_id: str
    comment_id: Optional[str]
    doc_kind: str
    created_utc: Optional[float]
    author: Optional[str]
    text: str


def flatten_comments(
    comments: Iterable[dict],
    source_file: str,
    subreddit: str,
    post_id: str,
) -> List[DocumentRow]:
    rows: List[DocumentRow] = []

    for comment in comments:
        author = comment.get("author")
        text = (comment.get("text") or "").strip()
        comment_id = comment.get("comment_id")

        if text and author != "AutoModerator":
            rows.append(
                DocumentRow(
                    source_file=source_file,
                    subreddit=subreddit,
                    post_id=post_id,
                    comment_id=comment_id,
                    doc_kind="comment",
                    created_utc=comment.get("created_utc"),
                    author=author,
                    text=text,
                )
            )

        child_rows = flatten_comments(
            comment.get("replies", []), source_file, subreddit, post_id
        )
        rows.extend(child_rows)

    return rows


def load_documents_from_file(file_path: str) -> List[DocumentRow]:
    file_obj = Path(file_path)
    subreddit = file_obj.parent.name

    with open(file_obj, "r", encoding="utf-8") as f:
        posts = json.load(f)

    rows: List[DocumentRow] = []

    for post in posts:
        if not post.get("comments_fetched"):
            continue

        post_id = post.get("post_id")
        post_data = post.get("post_data", {})
        post_text = (post_data.get("post_description") or "").strip()

        if post_text:
            rows.append(
                DocumentRow(
                    source_file=str(file_obj),
                    subreddit=subreddit,
                    post_id=post_id,
                    comment_id=None,
                    doc_kind="post",
                    created_utc=post.get("created_utc"),
                    author=None,
                    text=post_text,
                )
            )

        rows.extend(
            flatten_comments(
                post_data.get("comments", []),
                source_file=str(file_obj),
                subreddit=subreddit,
                post_id=post_id,
            )
        )

    return rows


def load_all_documents(input_files: List[str]) -> pd.DataFrame:
    all_rows: List[DocumentRow] = []
    for file_path in input_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input data file not found: {file_path}")
        all_rows.extend(load_documents_from_file(file_path))

    df = pd.DataFrame([r.__dict__ for r in all_rows])
    if df.empty:
        raise ValueError("No documents were loaded. Check input files and filters.")

    # Keep model inputs bounded but still informative for topic and emotion extraction.
    df["text_for_model"] = df["text"].str.slice(0, TEXT_MAX_CHARS)
    df["text_for_topic"] = df["text_for_model"].map(clean_topic_text)
    return df


def discover_data_files(data_glob: str) -> List[str]:
    files = sorted(glob.glob(data_glob))
    if not files:
        raise FileNotFoundError(f"No input data files found with glob: {data_glob}")
    return files


def clean_topic_text(text: str) -> str:
    cleaned = text.lower()
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"www\.\S+", " ", cleaned)
    cleaned = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", cleaned)
    cleaned = re.sub(r"&amp;", " and ", cleaned)
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = [tok for tok in cleaned.split() if tok not in NOISE_TERMS and len(tok) > 2]
    return " ".join(tokens)


def run_goemotions(df: pd.DataFrame) -> pd.DataFrame:
    device = 0 if torch.cuda.is_available() else -1
    classifier = pipeline(
        "text-classification",
        model=GOEMOTIONS_MODEL,
        top_k=None,
        truncation=True,
        batch_size=BATCH_SIZE,
        device=device,
    )

    unique_texts = pd.unique(df["text_for_model"])
    unique_predictions = classifier(unique_texts.tolist())
    prediction_lookup = dict(zip(unique_texts.tolist(), unique_predictions))

    emotion_rows: List[Dict[str, float]] = []
    top_emotions: List[str] = []
    top_scores: List[float] = []

    for text in df["text_for_model"]:
        pred = prediction_lookup[text]
        score_map = {item["label"]: float(item["score"]) for item in pred}
        emotion_rows.append(score_map)

        top_item = max(pred, key=lambda x: x["score"])
        top_emotions.append(top_item["label"])
        top_scores.append(float(top_item["score"]))

    emotions_df = pd.DataFrame(emotion_rows).fillna(0.0)
    out_df = pd.concat([df.reset_index(drop=True), emotions_df], axis=1)
    out_df["top_emotion"] = top_emotions
    out_df["top_emotion_score"] = top_scores
    return out_df


def run_bertopic(df: pd.DataFrame) -> Tuple[pd.DataFrame, BERTopic]:
    vectorizer_model = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        token_pattern=r"(?u)\b[a-z][a-z]+\b",
    )

    # Lower strictness to reduce the number of -1 outliers while keeping coherence.
    hdbscan_model = HDBSCAN(
        min_cluster_size=6,
        min_samples=2,
        cluster_selection_method="eom",
        prediction_data=True,
    )

    topic_model = BERTopic(
        embedding_model="all-MiniLM-L6-v2",
        vectorizer_model=vectorizer_model,
        hdbscan_model=hdbscan_model,
        calculate_probabilities=False,
        low_memory=True,
        min_topic_size=6,
        nr_topics=TARGET_NUM_TOPICS,
        verbose=True,
    )

    texts = df["text_for_topic"].tolist()
    topics, _ = topic_model.fit_transform(texts)

    out_df = df.copy()
    out_df["topic_id"] = topics

    topic_label_map = build_readable_topic_labels(topic_model)
    out_df["topic_label"] = out_df["topic_id"].map(topic_label_map).fillna("Unknown")

    out_df["topic_probability"] = np.nan
    return out_df, topic_model


def build_readable_topic_labels(topic_model: BERTopic) -> Dict[int, str]:
    label_map: Dict[int, str] = {-1: "Outlier/Mixed"}
    topic_info = topic_model.get_topic_info()

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        words = []
        for word, _score in (topic_model.get_topic(topic_id) or []):
            if word in NOISE_TERMS:
                continue
            if len(word) < 3:
                continue
            words.append(word)
            if len(words) == 3:
                break

        if words:
            label_map[topic_id] = " / ".join(words)
        else:
            label_map[topic_id] = f"Topic {topic_id}"

    return label_map


def load_absa_for_post(subreddit: str, post_id: str) -> Optional[dict]:
    absa_path = ABSA_ROOT / subreddit / f"{post_id}.json"
    if not absa_path.exists():
        return None

    with open(absa_path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_absa(df: pd.DataFrame) -> pd.DataFrame:
    ai_sentiments: List[Optional[str]] = []
    human_sentiments: List[Optional[str]] = []

    cache: Dict[Tuple[str, str], Optional[dict]] = {}

    for _, row in df.iterrows():
        key = (row["subreddit"], row["post_id"])
        if key not in cache:
            cache[key] = load_absa_for_post(row["subreddit"], row["post_id"])

        absa_data = cache[key]
        if absa_data is None:
            ai_sentiments.append(None)
            human_sentiments.append(None)
            continue

        if row["doc_kind"] == "post":
            post_absa = absa_data.get("post_absa", {})
            ai_sentiments.append(post_absa.get("ai"))
            human_sentiments.append(post_absa.get("human"))
            continue

        comment_id = row["comment_id"]
        comment_lookup = {
            c.get("comment_id"): c for c in absa_data.get("comments_absa", [])
        }
        comment_absa = comment_lookup.get(comment_id, {})
        ai_sentiments.append(comment_absa.get("ai"))
        human_sentiments.append(comment_absa.get("human"))

    out_df = df.copy()
    out_df["absa_ai_sentiment"] = ai_sentiments
    out_df["absa_human_sentiment"] = human_sentiments
    out_df["absa_ai_score"] = out_df["absa_ai_sentiment"].map(SENTIMENT_TO_SCORE)
    out_df["absa_human_score"] = out_df["absa_human_sentiment"].map(SENTIMENT_TO_SCORE)
    return out_df


def compute_topic_outputs(df: pd.DataFrame, topic_model: BERTopic) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    topic_summary = topic_model.get_topic_info().copy()

    topic_metrics = (
        df.groupby(["topic_id", "topic_label"], as_index=False)
        .agg(
            n_docs=("text", "count"),
            ai_sentiment_mean=("absa_ai_score", "mean"),
            human_sentiment_mean=("absa_human_score", "mean"),
            admiration_mean=("admiration", "mean"),
            gratitude_mean=("gratitude", "mean"),
            love_mean=("love", "mean"),
            sadness_mean=("sadness", "mean"),
            grief_mean=("grief", "mean"),
            nervousness_mean=("nervousness", "mean"),
        )
        .sort_values("n_docs", ascending=False)
    )

    topic_summary = topic_summary.merge(
        topic_metrics[["topic_id", "topic_label"]],
        how="left",
        left_on="Topic",
        right_on="topic_id",
    )
    topic_summary["topic_label"] = topic_summary["topic_label"].fillna(topic_summary["Name"])

    topic_dummies = pd.get_dummies(df["topic_label"], prefix="topic")
    corr_input = pd.concat(
        [
            topic_dummies,
            df[
                [
                    "absa_ai_score",
                    "absa_human_score",
                    "admiration",
                    "gratitude",
                    "love",
                    "sadness",
                    "grief",
                    "nervousness",
                ]
            ],
        ],
        axis=1,
    )

    corr_matrix = corr_input.corr(numeric_only=True)
    metric_cols = [
        "absa_ai_score",
        "absa_human_score",
        "admiration",
        "gratitude",
        "love",
        "sadness",
        "grief",
        "nervousness",
    ]
    topic_cols = [c for c in corr_matrix.index if c.startswith("topic_")]

    topic_corr = corr_matrix.loc[topic_cols, metric_cols].reset_index().rename(
        columns={"index": "topic_dummy"}
    )

    return topic_summary, topic_metrics, topic_corr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run BERTopic + GoEmotions + ABSA merge with ID-preserving outputs."
    )
    parser.add_argument(
        "--data-files",
        nargs="+",
        default=None,
        help="Input Reddit JSON files (posts_*.json).",
    )
    parser.add_argument(
        "--data-glob",
        default=DEFAULT_DATA_GLOB,
        help="Glob pattern for input Reddit JSON files when --data-files is omitted.",
    )
    parser.add_argument(
        "--absa-root",
        default=str(ABSA_ROOT),
        help="Directory containing ABSA outputs organized as results/absa/<subreddit>/<post_id>.json",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Directory to write merged tables and analysis outputs.",
    )
    return parser.parse_args()


def main():
    global ABSA_ROOT
    args = parse_args()

    ABSA_ROOT = Path(args.absa_root)
    data_files = args.data_files if args.data_files else discover_data_files(args.data_glob)
    output_root = Path(args.output_root)
    docs_csv = output_root / "documents_with_topics_emotions_absa.csv"
    topic_summary_csv = output_root / "topic_summary.csv"
    topic_metrics_csv = output_root / "topic_metrics.csv"
    topic_corr_csv = output_root / "topic_metric_correlations.csv"

    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading documents from source JSON files...")
    docs_df = load_all_documents(data_files)
    print(f"Loaded {len(docs_df)} post/comment documents.")

    print("Running GoEmotions classification...")
    docs_df = run_goemotions(docs_df)

    print("Running BERTopic topic modeling...")
    docs_df, topic_model = run_bertopic(docs_df)

    print("Merging ABSA results by post_id/comment_id...")
    docs_df = merge_absa(docs_df)

    print("Computing topic summaries and correlation tables...")
    topic_summary, topic_metrics, topic_corr = compute_topic_outputs(docs_df, topic_model)

    docs_df.to_csv(docs_csv, index=False)
    topic_summary.to_csv(topic_summary_csv, index=False)
    topic_metrics.to_csv(topic_metrics_csv, index=False)
    topic_corr.to_csv(topic_corr_csv, index=False)

    print("Done.")
    print(f"Saved document-level output to: {docs_csv}")
    print(f"Saved topic summary to: {topic_summary_csv}")
    print(f"Saved topic metrics to: {topic_metrics_csv}")
    print(f"Saved topic correlations to: {topic_corr_csv}")


if __name__ == "__main__":
    main()
