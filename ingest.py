import pdfplumber
from google import genai
import os
from dotenv import load_dotenv
from gemini_client import client, embed_text
import chromadb
import hashlib

# Load environment variables from the .env file
with pdfplumber.open("data/company_handbook_vn.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

# print(full_text)

# Function to split text into smaller chunks for embedding
def chunk_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(full_text)
# print(len(chunks))

# Load environment variables and initialize the Gemini API client
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# def embed_text(text):
#     result = client.models.embed_content(
#         model="gemini-embedding-001",
#         contents=text
#     )
#     return result.embeddings[0].values

# embeddings = [embed_text(chunk) for chunk in chunks]

# Function to generate a unique ID for each text chunk using SHA-256 hash
def get_chunk_id(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()

# Store the embeddings and corresponding text chunks in a persistent ChromaDB collection
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_handbook")

for chunk in chunks:
    chunk_id = get_chunk_id(chunk)
    
    existing = collection.get(ids=[chunk_id])
    if existing["ids"]:
        print(f"Chunk {chunk_id[:8]} đã tồn tại, bỏ qua embed")
        continue
    
    vector = embed_text(chunk)
    collection.add(ids=[chunk_id], embeddings=[vector], documents=[chunk])

print(f"Tổng số chunk trong collection: {collection.count()}")