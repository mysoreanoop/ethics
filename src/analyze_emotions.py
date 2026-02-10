import pandas as pd
from transformers import pipeline
import os

INPUT_FILE = "data/raw_data.csv"
OUTPUT_FILE = "data/analyzed_data.csv"

def analyze_emotions():
    print("Loading AI Model (this may take a moment)...")
    
    # FIX: Use 'top_k=None' to ensure we get scores for ALL emotions, not just the top one.
    classifier = pipeline(
        "text-classification", 
        model="bhadresh-savani/distilbert-base-uncased-emotion", 
        top_k=None
    )

    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Run 'make fetch' first.")
        return

    df = pd.read_csv(INPUT_FILE)
    
    # Simple text cleaning
    df['text'] = df['text'].fillna("").astype(str)
    # Filter out very short texts which might crash the model or give noise
    df = df[df['text'].str.len() > 5]

    print(f"Analyzing {len(df)} text items...")
    
    results = []
    
    for text in df['text']:
        # Truncate to 512 characters to fit model limits
        # Note: We create a list [text[:512]] because passing a single string sometimes 
        # changes the output format in different library versions.
        truncated_text = text[:512]
        
        try:
            # The model returns a list of lists: [[{'label': 'sadness', 'score': 0.1}, ...]]
            # We take the first element [0] to get the list for our single input.
            prediction = classifier(truncated_text)[0]
            
            # Flatten the scores into a simple dictionary: {'sadness': 0.01, 'joy': 0.99, ...}
            row_scores = {item['label']: item['score'] for item in prediction}
            results.append(row_scores)
            
        except Exception as e:
            print(f"Skipping row due to error: {e}")
            # Append empty scores to keep dataframe alignment
            results.append({})

    # Create a DataFrame from the results
    emotions_df = pd.DataFrame(results)
    
    # Combine original data with emotion scores
    # We use reset_index to ensure the rows align perfectly
    final_df = pd.concat([df.reset_index(drop=True), emotions_df], axis=1)

    # Calculate "Emotional Reliance" 
    # Ensure these columns exist before summing (fills NaN with 0 if missing)
    for col in ['love', 'sadness', 'fear']:
        if col not in final_df.columns:
            final_df[col] = 0.0

    final_df['reliance_score'] = (final_df['love'] + final_df['sadness'] + final_df['fear']) / 3

    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Analysis complete. Saved {len(final_df)} rows to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_emotions()
