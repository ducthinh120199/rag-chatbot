import psycopg2
import os
from datetime import datetime
from gemini_client import embed_batch
import chromadb
import time
import ollama
import os
from dotenv import load_dotenv
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LAST_SYNC_FILE = "last_sync.txt"
PIPELINE_VERSION = "v1_track_album_join"
BATCH_SIZE = 20
SLEEP_BETWEEN_BATCHES = 15

def get_last_sync_time():
    if os.path.exists(LAST_SYNC_FILE):
        with open(LAST_SYNC_FILE, "r") as f:
            return f.read().strip()
    return "1970-01-01 00:00:00"

def save_sync_time(timestamp):
    with open(LAST_SYNC_FILE, "w") as f:
        f.write(str(timestamp))

def embed_batch_local(texts):
    return [ollama.embed(model="nomic-embed-text", input=t)["embeddings"][0] for t in texts]

conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

last_sync = get_last_sync_time()
sync_start_time = datetime.now()

cursor.execute("""
    SELECT t.track_id, t.name, t.composer, t.unit_price, t.milliseconds,
           a.title AS album_title, g.name AS genre_name,
           t.is_deleted
    FROM track t
    LEFT JOIN album a ON t.album_id = a.album_id
    LEFT JOIN genre g ON t.genre_id = g.genre_id
    WHERE t.updated_at > %s
""", (last_sync,))

rows = cursor.fetchall()
print(f"Tìm thấy {len(rows)} bản ghi thay đổi kể từ {last_sync}")

chroma_client = chromadb.PersistentClient(path="./chroma_db_db")
collection = chroma_client.get_or_create_collection(name="chinook_tracks")

def process_batch(batch_rows):
    texts = []
    valid_rows = []
    for track_id, name, composer, unit_price, milliseconds, album_title, genre_name, is_deleted in batch_rows:
        if is_deleted:
            collection.delete(ids=[f"track_{track_id}"])
            continue
        text = f"Bài hát: {name}. Tác giả: {composer or 'Không rõ'}. Thuộc album: {album_title or 'Không rõ'}. Thể loại: {genre_name or 'Không rõ'}."
        texts.append(text)
        valid_rows.append((track_id, text, unit_price, milliseconds, album_title, genre_name))

    if not texts:
        return

    vectors = embed_batch(texts)

    for (track_id, text, unit_price, milliseconds, album_title, genre_name), vector in zip(valid_rows, vectors):
        collection.upsert(
            ids=[f"track_{track_id}"],
            embeddings=[vector],
            documents=[text],
            metadatas=[{
                "track_id": track_id, "album_title": album_title or "",
                "genre_name": genre_name or "", "unit_price": float(unit_price),
                "milliseconds": milliseconds, "pipeline_version": PIPELINE_VERSION
            }]
        )
    print(f"Đã xử lý batch {len(valid_rows)} dòng")

for i in range(0, len(rows), BATCH_SIZE):
    batch = rows[i:i + BATCH_SIZE]
    process_batch(batch)
    time.sleep(15)

save_sync_time(sync_start_time)
cursor.close()
conn.close()