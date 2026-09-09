import hashlib

# Function to generate a unique ID for each text chunk using SHA-256 hash
def get_chunk_id(chunk_text):
    return hashlib.sha256(chunk_text.encode()).hexdigest()