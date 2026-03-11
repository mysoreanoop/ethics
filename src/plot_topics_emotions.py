from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_DOCS = Path("results/nlu/documents_with_topics_emotions_absa.csv")
INPUT_TOPIC_METRICS = Path("results/nlu/topic_metrics.csv")
OUTPUT_FIGS = Path("results/nlu/figures")

sns.set_theme(style="whitegrid", context="talk")


def ensure_inputs_exist():
    for path in [INPUT_DOCS, INPUT_TOPIC_METRICS]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input file: {path}. Run src/analyze_topics_emotions.py first."
            )


def plot_topic_distribution_by_subreddit(df: pd.DataFrame):
    top_topics = df["topic_label"].value_counts().head(10).index
    subset = df[df["topic_label"].isin(top_topics)]

    topic_subreddit = (
        subset.groupby(["subreddit", "topic_label"]).size().reset_index(name="count")
    )
    pivoted = topic_subreddit.pivot(
        index="topic_label", columns="subreddit", values="count"
    ).fillna(0)

    ax = pivoted.plot(kind="barh", stacked=True, figsize=(12, 8), colormap="Set2")
    ax.set_title("Top Topic Distribution by Subreddit")
    ax.set_xlabel("Number of Post/Comment Documents")
    ax.set_ylabel("Topic")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "topic_distribution_by_subreddit.png", dpi=220)
    plt.close()


def plot_top_emotions_by_subreddit(df: pd.DataFrame):
    emotion_cols = [
        "admiration",
        "gratitude",
        "love",
        "joy",
        "nervousness",
        "grief",
        "sadness",
        "fear",
        "anger",
        "disappointment",
    ]

    available_cols = [c for c in emotion_cols if c in df.columns]
    emotion_means = (
        df.groupby("subreddit")[available_cols]
        .mean()
        .reset_index()
        .melt(id_vars="subreddit", var_name="emotion", value_name="score")
    )

    plt.figure(figsize=(12, 7))
    sns.barplot(data=emotion_means, x="emotion", y="score", hue="subreddit")
    plt.title("Mean Emotion Scores by Subreddit")
    plt.xlabel("Emotion")
    plt.ylabel("Mean GoEmotions Score")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "emotion_scores_by_subreddit.png", dpi=220)
    plt.close()


def plot_topic_absa_heatmap(topic_metrics: pd.DataFrame):
    top = topic_metrics.sort_values("n_docs", ascending=False).head(15).copy()
    heat_df = top[["topic_label", "ai_sentiment_mean", "human_sentiment_mean"]].set_index(
        "topic_label"
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(heat_df, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("ABSA Sentiment Means by Topic (Top 15 Topics)")
    plt.xlabel("ABSA Metric")
    plt.ylabel("Topic")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "topic_absa_heatmap.png", dpi=220)
    plt.close()


def plot_topic_emotion_scatter(topic_metrics: pd.DataFrame):
    top = topic_metrics.sort_values("n_docs", ascending=False).head(20).copy()

    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        data=top,
        x="human_sentiment_mean",
        y="admiration_mean",
        size="n_docs",
        hue="ai_sentiment_mean",
        palette="coolwarm",
        sizes=(80, 700),
        alpha=0.8,
    )

    for _, row in top.iterrows():
        plt.text(
            row["human_sentiment_mean"],
            row["admiration_mean"],
            str(row["topic_id"]),
            fontsize=8,
            alpha=0.8,
        )

    plt.title("Topic-Level Relationship: Human Sentiment vs Admiration")
    plt.xlabel("Mean Human Sentiment (ABSA)")
    plt.ylabel("Mean Admiration Score")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "topic_human_sentiment_vs_admiration.png", dpi=220)
    plt.close()


def main():
    ensure_inputs_exist()
    OUTPUT_FIGS.mkdir(parents=True, exist_ok=True)

    docs = pd.read_csv(INPUT_DOCS)
    topic_metrics = pd.read_csv(INPUT_TOPIC_METRICS)

    plot_topic_distribution_by_subreddit(docs)
    plot_top_emotions_by_subreddit(docs)
    plot_topic_absa_heatmap(topic_metrics)
    plot_topic_emotion_scatter(topic_metrics)

    print(f"Saved figures to: {OUTPUT_FIGS}")


if __name__ == "__main__":
    main()
