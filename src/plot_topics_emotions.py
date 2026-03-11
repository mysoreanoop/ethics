from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_DOCS = Path("results/nlu/documents_with_topics_emotions_absa.csv")
INPUT_TOPIC_METRICS = Path("results/nlu/topic_metrics.csv")
OUTPUT_FIGS = Path("results/nlu/figures")

GOEMOTIONS_LABELS = [
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
]

sns.set_theme(style="whitegrid", context="talk")


def ensure_inputs_exist():
    for path in [INPUT_DOCS, INPUT_TOPIC_METRICS]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing required input file: {path}. Run src/analyze_topics_emotions.py first."
            )


def plot_topic_distribution_by_subreddit(df: pd.DataFrame):
    posts_only = df[df["doc_kind"] == "post"]
    top_topics = posts_only["topic_label"].value_counts().head(20).index
    subset = posts_only[posts_only["topic_label"].isin(top_topics)]

    topic_subreddit = (
        subset.groupby(["subreddit", "topic_label"]).size().reset_index(name="count")
    )
    pivoted = topic_subreddit.pivot(
        index="topic_label", columns="subreddit", values="count"
    ).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 6))
    pivoted.plot(kind="barh", stacked=True, colormap="Set2", ax=ax)
    ax.set_title("Top 20 Topic Distribution by Subreddit (Posts Only)")
    ax.set_xlabel("Number of Posts")
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=8, pad=1)
    ax.margins(y=0.01)
    plt.subplots_adjust(left=0.30, right=0.97, top=0.92, bottom=0.10)
    plt.savefig(OUTPUT_FIGS / "topic_distribution_by_subreddit.png", dpi=220)
    plt.close()


def plot_top_emotions_by_subreddit(df: pd.DataFrame):
    available_cols = [c for c in GOEMOTIONS_LABELS if c in df.columns]
    emotion_means = (
        df.groupby("subreddit")[available_cols]
        .mean()
        .reset_index()
        .melt(id_vars="subreddit", var_name="emotion", value_name="score")
    )

    plt.figure(figsize=(18, 8))
    sns.barplot(data=emotion_means, x="emotion", y="score", hue="subreddit")
    plt.title("Mean GoEmotions Scores by Subreddit (All 28 Labels)")
    plt.xlabel("Emotion")
    plt.ylabel("Mean GoEmotions Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGS / "emotion_scores_by_subreddit.png", dpi=220)
    plt.close()


def plot_topic_absa_heatmap(topic_metrics: pd.DataFrame):
    top = topic_metrics[topic_metrics["topic_id"] != -1].sort_values(
        "n_docs", ascending=False
    ).head(15).copy()
    heat_df = top[["topic_label", "ai_sentiment_mean", "human_sentiment_mean"]].set_index(
        "topic_label"
    )
    heat_df.columns = ["Toward AI", "Toward Humans"]

    fig, ax = plt.subplots(figsize=(8, 10))
    sns.heatmap(
        heat_df,
        annot=True,
        cmap="coolwarm",
        center=0,
        fmt=".2f",
        annot_kws={"size": 11},
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("ABSA Sentiment by Topic (Top 15)", pad=14)
    ax.set_xlabel("Sentiment Toward", labelpad=10)
    ax.set_ylabel("")
    ax.tick_params(axis="y", labelsize=9)
    ax.tick_params(axis="x", labelsize=11)
    plt.subplots_adjust(left=0.38, right=0.97, top=0.93, bottom=0.08)
    plt.savefig(OUTPUT_FIGS / "topic_absa_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()


def plot_topic_emotion_scatter(topic_metrics: pd.DataFrame):
    top = topic_metrics.sort_values("n_docs", ascending=False).head(20).copy()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.scatterplot(
        data=top,
        x="human_sentiment_mean",
        y="admiration_mean",
        size="n_docs",
        hue="ai_sentiment_mean",
        palette="coolwarm",
        sizes=(80, 700),
        alpha=0.8,
        ax=ax,
    )

    for _, row in top.iterrows():
        ax.text(
            row["human_sentiment_mean"],
            row["admiration_mean"],
            str(row["topic_id"]),
            fontsize=8,
            alpha=0.8,
        )

    ax.set_title("Topic-Level Relationship: Human Sentiment vs Admiration")
    ax.set_xlabel("Mean Human Sentiment (ABSA)")
    ax.set_ylabel("Mean Admiration Score")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0,
        frameon=True,
    )
    plt.subplots_adjust(right=0.75)
    plt.savefig(OUTPUT_FIGS / "topic_human_sentiment_vs_admiration.png", dpi=220, bbox_inches="tight")
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
