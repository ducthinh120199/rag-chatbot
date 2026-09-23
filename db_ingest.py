import psycopg2
import os
import time
from datetime import datetime
from gemini_client import embed_batch
from pgvector.psycopg2 import register_vector
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
LAST_SYNC_FILE = "last_sync.txt"
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


def process_batch(cursor, conn, batch_rows):
    texts = []
    valid_track_ids = []
    deleted_ids = []

    for track_id, name, composer, unit_price, milliseconds, album_title, genre_name, is_deleted in batch_rows:
        if is_deleted:
            deleted_ids.append(track_id)
            continue
        text = f"Bài hát: {name}. Tác giả: {composer or 'Không rõ'}. Thuộc album: {album_title or 'Không rõ'}. Thể loại: {genre_name or 'Không rõ'}."
        texts.append(text)
        valid_track_ids.append(track_id)

    if deleted_ids:
        cursor.execute("UPDATE track SET embedding = NULL WHERE track_id = ANY(%s)", (deleted_ids,))
        conn.commit()
        print(f"Đã xóa embedding cho {len(deleted_ids)} dòng bị đánh dấu is_deleted")

    if not texts:
        return

    vectors = embed_batch(texts)

    for track_id, vector in zip(valid_track_ids, vectors):
        cursor.execute("UPDATE track SET embedding = %s WHERE track_id = %s", (vector, track_id))
    conn.commit()
    print(f"Đã cập nhật embedding cho {len(valid_track_ids)} dòng")


def main():
    conn = psycopg2.connect(DATABASE_URL)
    register_vector(conn)
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

    for i in range(0, len(rows), BATCH_SIZE):
        batch = rows[i:i + BATCH_SIZE]
        process_batch(cursor, conn, batch)
        time.sleep(SLEEP_BETWEEN_BATCHES)

    save_sync_time(sync_start_time)
    cursor.close()
    conn.close()


if __name__ == "__main__":
    main()