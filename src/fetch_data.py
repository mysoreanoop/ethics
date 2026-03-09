import requests
import pandas as pd
import time
import os

# CONFIG
SUBREDDIT = "MyGirlfriendIsAI"  # Change to "CharacterAI" or "Replika" if empty
LIMIT = 100  # Number of posts to fetch
OUTPUT_FILE = "data/raw_data.csv"

HEADERS = {"User-Agent": "Mozilla/5.0 (Script/1.0)"}

def fetch_posts():
    url = f"https://www.reddit.com/r/{SUBREDDIT}/new.json?limit={LIMIT}"
    response = requests.get(url, headers=HEADERS)
    
    if response.status_code != 200:
        print(f"Failed to fetch posts: {response.status_code}")
        return []

    posts_data = []
    children = response.json()['data']['children']
    
    print(f"Found {len(children)} posts. Fetching comments...")

    for post in children:
        p_data = post['data']
        post_id = p_data['id']
        
        # Add the post itself
        posts_data.append({
            "id": post_id,
            "type": "post",
            "text": p_data['title'] + " " + p_data['selftext'],
            "score": p_data['score'],
            "timestamp": p_data['created_utc']
        })

        # Fetch comments for this post
        # (Note: Reddit JSON adds a trailing slash for comments)
        comm_url = f"https://www.reddit.com{p_data['permalink']}.json"
        time.sleep(2)  # Polite delay
        
        try:
            comm_resp = requests.get(comm_url, headers=HEADERS)
            if comm_resp.status_code == 200:
                # Comments are in the second element of the list
                comments = comm_resp.json()[1]['data']['children']
                for comm in comments[:5]: # Limit to top 5 comments per post
                    if 'body' in comm['data']:
                        posts_data.append({
                            "id": post_id, # Link back to parent post
                            "type": "comment",
                            "text": comm['data']['body'],
                            "score": comm['data']['score'],
                            "timestamp": comm['data']['created_utc']
                        })
        except Exception as e:
            print(f"Error fetching comments for {post_id}: {e}")

    return pd.DataFrame(posts_data)

if __name__ == "__main__":
    df = fetch_posts()
    if not df.empty:
        os.makedirs("data", exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved {len(df)} items to {OUTPUT_FILE}")
    else:
        print("No data found.")
