import requests
import time
import os
import json
from datetime import datetime
import glob

SUBREDDIT = "MyGirlfriendIsAI"
LIMIT = 100
MAX_POSTS = 1000
DATA_DIR = "data"

HEADERS = {"User-Agent": "Mozilla/5.0 (Script/1.0)"}

def get_existing_post_ids():
    existing_ids = set()
    if not os.path.exists(DATA_DIR):
        return existing_ids
    
    for filepath in glob.glob(os.path.join(DATA_DIR, "posts_*.json")):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                for item in data:
                    existing_ids.add(item.get("post_id"))
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return existing_ids

def fetch_and_save_posts():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_ids = get_existing_post_ids()
    print(f"Found {len(existing_ids)} existing posts.")
    
    after = None
    total_fetched = 0
    
    while total_fetched < MAX_POSTS:
        url = f"https://www.reddit.com/r/{SUBREDDIT}/new.json?limit={LIMIT}"
        if after:
            url += f"&after={after}"
            
        print(f"Fetching posts... (after: {after})")
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code != 200:
            print(f"Failed to fetch posts: {response.status_code}")
            break
            
        json_data = response.json()
        children = json_data.get('data', {}).get('children', [])
        
        if not children:
            print("No more posts found.")
            break
            
        batch_data = []
        new_in_batch = 0
        
        for post in children:
            p_data = post['data']
            post_id = p_data['id']
            
            if post_id in existing_ids:
                continue
                
            existing_ids.add(post_id)
            new_in_batch += 1
            
            # Store initial data, prepare for get_comments
            batch_data.append({
                "post_id": post_id,
                "permalink": p_data.get('permalink', ''),
                "title": p_data.get('title', ''),
                "created_utc": p_data.get('created_utc', ''),
                "comments_fetched": False,
                "post_data": {}
            })
            
        if batch_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(DATA_DIR, f"posts_{timestamp}.json")
            with open(filename, "w", encoding="utf-8") as f:
                json.dump(batch_data, f, indent=4)
            print(f"Saved {len(batch_data)} new posts to {filename}")
        else:
            print("No new posts in this batch.")
            
        # We count all fetched posts (even skipped duplicates) to avoid infinite pagination loops
        total_fetched += len(children)
        after = json_data.get('data', {}).get('after')
        
        if not after:
            print("Reached end of pagination.")
            break
            
        if total_fetched < MAX_POSTS:
            time.sleep(2)

if __name__ == "__main__":
    fetch_and_save_posts()
