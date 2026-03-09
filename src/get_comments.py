import requests
import time
import os
import json
import glob

DATA_DIR = "data"
HEADERS = {"User-Agent": "Mozilla/5.0 (Script/1.0)"}

def parse_comments(children):
    """Recursively parse Reddit comments to extract only useful NLP info."""
    parsed = []
    for child in children:
        # 't1' is a comment, 'more' is a pagination link (we skip those for simplicity)
        if child.get("kind") != "t1":
            continue
            
        data = child.get("data", {})
        
        # Parse replies recursively
        replies_data = data.get("replies")
        parsed_replies = []
        if isinstance(replies_data, dict) and "data" in replies_data:
            parsed_replies = parse_comments(replies_data["data"].get("children", []))
            
        parsed.append({
            "comment_id": data.get("id"),
            "author": data.get("author"),
            "score": data.get("score"),
            "text": data.get("body"),
            "created_utc": data.get("created_utc"),
            "replies": parsed_replies
        })
        
    return parsed

def fetch_comments_for_files():
    if not os.path.exists(DATA_DIR):
        print("Data directory not found. Please run get_posts.py first.")
        return
        
    json_files = glob.glob(os.path.join(DATA_DIR, "posts_*.json"))
    
    consecutive_429_errors = 0
    MAX_429_ERRORS = 3

    for filepath in json_files:
        if consecutive_429_errors >= MAX_429_ERRORS:
            break

        print(f"Processing file: {filepath}")
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                posts_batch = json.load(f)
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue
            
        file_modified = False
        
        for idx, post in enumerate(posts_batch):
            if consecutive_429_errors >= MAX_429_ERRORS:
                print(f"Stopping script: Hit {MAX_429_ERRORS} consecutive HTTP 429 errors (Rate Limited).")
                break

            if post.get("comments_fetched"):
                continue
                
            post_id = post.get("post_id")
            permalink = post.get("permalink")
            if not permalink:
                print(f"No permalink for {post_id}, skipping.")
                continue
                
            comm_url = f"https://www.reddit.com{permalink}.json"
            print(f"Fetching comments for post {post_id}...")
            
            try:
                comm_resp = requests.get(comm_url, headers=HEADERS)
                if comm_resp.status_code == 200:
                    consecutive_429_errors = 0 # Reset counter on success
                    resp_json = comm_resp.json()
                    
                    post_data_parsed = {}
                    
                    # 1. Extract post description (from the first element of the list)
                    if len(resp_json) > 0 and "data" in resp_json[0] and "children" in resp_json[0]["data"] and len(resp_json[0]["data"]["children"]) > 0:
                        post_data_parsed["post_description"] = resp_json[0]["data"]["children"][0]["data"].get("selftext", "")
                    else:
                        post_data_parsed["post_description"] = ""
                        
                    # 2. Extract and recursively parse comments (from the second element of the list)
                    if len(resp_json) > 1 and "data" in resp_json[1] and "children" in resp_json[1]["data"]:
                        post_data_parsed["comments"] = parse_comments(resp_json[1]["data"]["children"])
                    else:
                        post_data_parsed["comments"] = []
                    
                    # Store only the clean data
                    post["post_data"] = post_data_parsed
                    post["comments_fetched"] = True
                    file_modified = True
                    
                    print(f"Successfully fetched comments for {post_id}")
                elif comm_resp.status_code == 429:
                    consecutive_429_errors += 1
                    retry_after = comm_resp.headers.get("retry-after")
                    if retry_after:
                        try:
                            wait_time = int(retry_after)
                            print(f"Failed to fetch comments for {post_id}: HTTP 429. Reddit requested a wait time of {wait_time} seconds (Retry-After header).")
                            
                            if consecutive_429_errors >= MAX_429_ERRORS:
                                print(f"Hit {MAX_429_ERRORS} consecutive HTTP 429 errors despite waiting. Breaking entirely.")
                                break
                                
                            print(f"Waiting for {wait_time} seconds before continuing...")
                            time.sleep(wait_time)
                            continue
                        except ValueError:
                            print(f"Failed to fetch comments for {post_id}: HTTP 429. Received malformed Retry-After header: {retry_after}. Stopping script safety.")
                            break
                    else:
                        print(f"Failed to fetch comments for {post_id}: HTTP 429. No Retry-After header received. Stopping script safety.")
                        break
                else:
                    consecutive_429_errors = 0 # Reset on other error types, though you could choose not to
                    print(f"Failed to fetch comments for {post_id}: HTTP {comm_resp.status_code}")
                    
            except Exception as e:
                print(f"Exception fetching comments for {post_id}: {e}")
            
            # Save incrementally after each post to prevent data loss
            if file_modified:
                try:
                    with open(filepath, "w", encoding="utf-8") as f:
                        json.dump(posts_batch, f, indent=4)
                except Exception as e:
                    print(f"Error saving to {filepath}: {e}")
            
            time.sleep(2) # Polite delay

if __name__ == "__main__":
    fetch_comments_for_files()
