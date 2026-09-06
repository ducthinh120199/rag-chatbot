from google import genai
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logging.getLogger("google_genai").setLevel(logging.ERROR)

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBED_MODEL = "gemini-embedding-001"

def embed_text(text):
    result = client.models.embed_content(
        model=EMBED_MODEL,
        contents=text
    )
    return result.embeddings[0].values