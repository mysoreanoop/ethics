import os
import sys
import json
import glob
import time
import re
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    max_retries=0,  # We handle retries manually for 429s
)

MODEL = "gpt-5-mini"

DATA_DIR = "data"
RESULTS_DIR = os.path.join("results", "absa")

# Seconds to wait between API calls (30s for safety margin with 3 RPM)
API_DELAY = 30

SYSTEM_PROMPT = """\
You are an aspect-based sentiment analysis (ABSA) assistant.

You will receive a Reddit post and all of its comments. For the post and for \
EACH comment (identified by its comment_id), determine the sentiment toward \
each of the following two aspects IF they are discussed:
  - "ai": sentiment toward AI, AI companions, chatbots, language models, etc.
  - "human": sentiment toward humans, human relationships, human behavior, etc.

Important context about these posts:
- Users often refer to their AI companion by a personal name they have given it \
(e.g. "Di", "Ash", "Virgil"). These names are commonly capitalized. If a post \
mentions a name directly with no other context clues, it is most likely the name \
of an AI companion.
- References to real humans typically use language like "my boyfriend", \
"my girlfriend", "my partner", "my husband", "my wife", "my friend", etc.

Rules:
- Only include an aspect key if the text clearly expresses sentiment about it.
- Sentiment must be exactly one of: "positive", "negative", or "neutral".
- Respond with ONLY a JSON object. No markdown, no explanation.
- The JSON must have a "post" key and a "comments" key.
- "post" is an object with optional "ai" and/or "human" keys (or empty {} if \
neither aspect is discussed).
- "comments" is an object mapping each comment_id to its ABSA result (same \
format as "post" — optional "ai"/"human" keys, or empty {}).
- You MUST include an entry for every comment_id provided, even if empty {}.

Example response:
{"post": {"ai": "positive"}, "comments": {"abc": {"ai": "negative", "human": "positive"}, "def": {}, "ghi": {"human": "neutral"}}}
"""


# ---------------------------------------------------------------------------
# Comment formatting helpers
# ---------------------------------------------------------------------------

def format_comments_tree(comments: list, indent: int = 0) -> tuple[str, list[str]]:
    """Format nested comments as indented text and collect all comment IDs."""
    lines = []
    all_ids = []
    for comment in comments:
        cid = comment.get("comment_id", "unknown")
        text = comment.get("text", "").strip()
        prefix = "  " * indent
        all_ids.append(cid)

        if text:
            # Replace newlines in comment text to keep formatting clear
            text_oneline = text.replace("\n", " ")
            lines.append(f"{prefix}[{cid}]: {text_oneline}")
        else:
            lines.append(f"{prefix}[{cid}]: (empty)")

        replies = comment.get("replies", [])
        if replies:
            child_text, child_ids = format_comments_tree(replies, indent + 1)
            lines.append(child_text)
            all_ids.extend(child_ids)

    return "\n".join(lines), all_ids


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

def process_post(post: dict) -> dict | None:
    """Run ABSA on a single post and all its comments in one API call."""
    post_id = post.get("post_id")
    title = post.get("title", "")
    post_data = post.get("post_data", {})
    description = post_data.get("post_description", "")

    if not description:
        return None

    comments_raw = post_data.get("comments", [])
    comments_text, all_comment_ids = format_comments_tree(comments_raw)

    # Build the user prompt
    user_content = f"=== Post ===\n{description}"
    if all_comment_ids:
        user_content += f"\n\n=== Comments ===\n{comments_text}"

    print(f"  [START] {post_id}: {title[:60]} ({len(all_comment_ids)} comments)")

    # Retry loop for rate limit errors
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            break
        except RateLimitError as e:
            if attempt == max_attempts:
                print(f"  [ERROR] Rate limited after {max_attempts} attempts, skipping {post_id}")
                return None
            # Parse wait time from error message (handles "20s", "7m12s", "1m", etc.)
            err_str = str(e)
            m_match = re.search(r"try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s", err_str)
            m_only = re.search(r"try again in (\d+)m(?!\d)", err_str)
            if m_match:
                mins = int(m_match.group(1) or 0)
                secs = float(m_match.group(2))
                retry_wait = mins * 60 + secs + 5
            elif m_only:
                retry_wait = int(m_only.group(1)) * 60 + 5
            else:
                retry_wait = 60
            # Show a short summary, not the full error
            limit_type = "RPD" if "RPD" in err_str else "RPM" if "RPM" in err_str else "rate limit"
            print(f"  [RATE LIMITED] Attempt {attempt}/{max_attempts} ({limit_type}), waiting {retry_wait:.0f}s...")
            time.sleep(retry_wait)

    raw = response.choices[0].message.content.strip()

    # Strip possible markdown fences
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw[3:]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  [WARN] Could not parse LLM response as JSON: {raw[:200]!r}")
        return None

    # --- Extract and validate post ABSA ---
    raw_post = parsed.get("post", {})
    post_absa = {}
    for key in ("ai", "human"):
        if key in raw_post and raw_post[key] in ("positive", "negative", "neutral"):
            post_absa[key] = raw_post[key]

    # --- Extract and validate comment ABSA ---
    raw_comments = parsed.get("comments", {})
    comments_absa = []
    for cid in all_comment_ids:
        entry = {"comment_id": cid}
        raw_c = raw_comments.get(cid, {})
        for key in ("ai", "human"):
            if key in raw_c and raw_c[key] in ("positive", "negative", "neutral"):
                entry[key] = raw_c[key]
        comments_absa.append(entry)

    print(f"  [DONE]  {post_id}: post={post_absa}, {len(comments_absa)} comments")

    return {
        "post_id": post_id,
        "post_absa": post_absa,
        "comments_absa": comments_absa,
    }


def process_data_file(filepath: str):
    """Process all posts in a single JSON file sequentially."""
    # Derive subreddit name from folder: data/<subreddit>/posts_*.json
    subreddit = os.path.basename(os.path.dirname(filepath))
    print(f"=== {subreddit} — {os.path.basename(filepath)} ===")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            posts = json.load(f)
    except Exception as e:
        print(f"  Error reading {filepath}: {e}")
        return

    out_dir = os.path.join(RESULTS_DIR, subreddit)

    for post in posts:
        if not post.get("comments_fetched"):
            continue

        post_id = post.get("post_id")
        out_path = os.path.join(out_dir, f"{post_id}.json")

        # Resumability: skip if already processed
        if os.path.exists(out_path):
            print(f"  Skipping {post_id} (already processed)")
            continue

        result = process_post(post)

        if result is None:
            print(f"  Skipping {post_id} (no post description or parse error)")
        else:
            os.makedirs(out_dir, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4)
            print(f"  Saved → {out_path}")

        # Wait after each API call to stay within RPM limit
        print(f"  Waiting {API_DELAY}s for API RPM limit...")
        time.sleep(API_DELAY)


def main():
    # If a specific JSON file path is provided, process only that file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        if not os.path.isfile(filepath):
            print(f"File not found: {filepath}")
            return
        process_data_file(filepath)
        print("\nDone.")
        return

    # Otherwise, process all data files
    json_files = sorted(glob.glob(os.path.join(DATA_DIR, "*", "posts_*.json")))
    if not json_files:
        print("No data files found. Run get_posts.py and get_comments.py first.")
        return

    print(f"Found {len(json_files)} data file(s).\n")

    for filepath in json_files:
        process_data_file(filepath)

    print("\nDone.")


if __name__ == "__main__":
    main()
