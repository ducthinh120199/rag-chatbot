from google import genai
import os
from dotenv import load_dotenv
import time
from google.genai import errors as genai_errors
import logging

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

load_dotenv()
logging.getLogger("google_genai").setLevel(logging.ERROR)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"

def embed_text(text, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = client.models.embed_content(
                model=EMBED_MODEL,
                contents=text
            )
            return result.embeddings[0].values
        except genai_errors.ClientError as e:
            if e.code not in RETRYABLE_STATUS_CODES:
                raise
            if attempt == max_retries - 1:
                raise
            wait_time = 2 ** attempt
            print(f"Lỗi tạm thời ({e.code} {e.status}), thử lại sau {wait_time}s...")
            time.sleep(wait_time)