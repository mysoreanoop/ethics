import requests
import json
import time

def get_posts_without_login(sub_name, limit=10):
    # 1. The URL adds .json to the subreddit link
    url = f"https://www.reddit.com/r/{sub_name}/new.json?limit={limit}"
    
    # 2. CRITICAL: You must use a custom User-Agent or Reddit will block you
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    try:
        response = requests.get(url, headers=headers)
        
        # Check if successful
        if response.status_code == 200:
            data = response.json()
            posts = data['data']['children']
            
            print(f"--- Found {len(posts)} posts in r/{sub_name} ---\n")
            
            for post in posts:
                post_data = post['data']
                print(f"Title:  {post_data['title']}")
                print(f"Author: {post_data['author']}")
                print(f"Upvotes: {post_data['score']}")
                print(f"Link:   https://reddit.com{post_data['permalink']}")
                print("-" * 40)
        
        elif response.status_code == 403:
            print("Error 403: Access Forbidden. The subreddit might be private.")
        elif response.status_code == 429:
            print("Error 429: Too Many Requests. Wait a moment and try again.")
        else:
            print(f"Error: Status Code {response.status_code}")

    except Exception as e:
        print(f"Script failed: {e}")

if __name__ == "__main__":
    get_posts_without_login("mygirlfriendisai", limit=5)
