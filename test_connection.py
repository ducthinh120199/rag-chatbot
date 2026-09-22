import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM track")
    result = cursor.fetchone()
    print(f"Kết nối thành công! Số dòng trong bảng track: {result[0]}")
    cursor.close()
    conn.close()
except Exception as e:
    print(f"LỖI kết nối: {e}")