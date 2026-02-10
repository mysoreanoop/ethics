import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

INPUT_FILE = "data/analyzed_data.csv"
PLOT_DIR = "plots"

def generate_plots():
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Overall Emotion Distribution
    plt.figure(figsize=(10, 6))
    emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
    avg_emotions = df[emotions].mean().sort_values(ascending=False)
    sns.barplot(x=avg_emotions.index, y=avg_emotions.values, palette="viridis")
    plt.title("Average Emotion Intensity in Subreddit")
    plt.ylabel("Confidence Score")
    plt.savefig(f"{PLOT_DIR}/emotion_distribution.png")
    
    # 2. Emotional Reliance: Posts vs Comments
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='type', y='reliance_score', data=df)
    plt.title("Emotional Reliance Score: Posts vs Comments")
    plt.savefig(f"{PLOT_DIR}/reliance_by_type.png")

    # 3. Correlation Matrix (Do highly scored posts show more sadness?)
    plt.figure(figsize=(10, 8))
    corr = df[['score', 'reliance_score'] + emotions].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation: Upvotes vs Emotions")
    plt.savefig(f"{PLOT_DIR}/correlation_matrix.png")

    print(f"Plots saved to {PLOT_DIR}/")

    # Simple Conclusion Output
    avg_reliance = df['reliance_score'].mean()
    print("\n--- CONCLUSIONS ---")
    print(f"Average 'Reliance' Score: {avg_reliance:.4f}")
    print("Top Emotion:", avg_emotions.idxmax())
    print("If 'sadness' or 'love' are dominant, it may indicate high emotional attachment.")

if __name__ == "__main__":
    generate_plots()
