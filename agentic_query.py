import psycopg2
import ollama
import chromadb
import re
import requests
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

ALLOWED_TABLES = {"track", "album", "genre"}

SCHEMA_INFO = """
Bảng track(track_id, name, composer, album_id, genre_id, unit_price, milliseconds)
Bảng album(album_id, title, artist_id)
Bảng genre(genre_id, name)

QUAN TRỌNG: Đây là PostgreSQL. Dùng dấu nháy ĐƠN '...' cho giá trị chuỗi (ví dụ: WHERE title = 'ABC').
KHÔNG dùng dấu nháy kép "..." cho giá trị chuỗi — dấu nháy kép chỉ dùng cho tên cột/bảng.
Chỉ viết ĐÚNG 1 câu SELECT, không giải thích gì thêm, không dùng markdown code block.
Ví dụ câu SQL đúng: SELECT t.name FROM track t JOIN album a ON t.album_id = a.album_id WHERE a.title = 'ABC';
"""

def choose_tool_with_jev(question):
    response = requests.post(
        "https://ai-gateway.vercel.sh/typesafe/v1/systemone",
        headers={
            "Authorization": f"Bearer {os.getenv('AI_GATEWAY_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "typesafe-ai/jev",
            "state": question,
            "questions": {
                "tool_choice": {
                    "type": "choice",
                    "criteria": {
                        "semantic_search": "Dùng cho câu hỏi mô tả, mơ hồ, không cần liệt kê chính xác toàn bộ",
                        "run_sql_query": "Dùng khi câu hỏi cần liệt kê TOÀN BỘ, ĐẾM CHÍNH XÁC, hoặc lọc theo điều kiện rõ ràng"
                    },
                    "instructions": "Chọn công cụ phù hợp nhất để trả lời câu hỏi."
                }
            }
        }
    )
    result = response.json()
    answer = result["answers"]["tool_choice"]
    print(f"[Jev chọn tool: {answer['choice']}, confidence: {answer['confidence']}]")
    return answer["choice"]

def semantic_search(query):
    vector = ollama.embed(model="nomic-embed-text", input=query)["embeddings"][0]
    chroma_client = chromadb.PersistentClient(path="./chroma_db_db")
    collection = chroma_client.get_or_create_collection(name="chinook_tracks")
    results = collection.query(query_embeddings=[vector], n_results=5, include=["documents"])
    return "\n".join(results["documents"][0])

def generate_sql_with_ollama(question):
    prompt = f"Schema database:\n{SCHEMA_INFO}\n\nViết câu SQL SELECT để trả lời câu hỏi sau: {question}"
    response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"].strip()

def is_sql_safe(sql_query):
    normalized = sql_query.strip().lower()
    normalized_no_trailing_semicolon = normalized.rstrip(";").strip()

    if not normalized_no_trailing_semicolon.startswith("select"):
        return False, "Chỉ cho phép câu lệnh SELECT."

    # Kiểm tra dấu ; ở giữa câu (dấu hiệu nối thêm lệnh khác) — bỏ qua 1 dấu ; cuối cùng nếu có
    if ";" in normalized_no_trailing_semicolon:
        return False, "Câu lệnh chứa dấu ';' ở giữa — nghi ngờ cố nối thêm lệnh khác."

    dangerous_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "--"]
    for keyword in dangerous_keywords:
        if keyword in normalized_no_trailing_semicolon:
            return False, f"Câu lệnh chứa từ khóa không được phép: '{keyword}'"

    tables_in_query = set(re.findall(r'\bfrom\s+(\w+)|\bjoin\s+(\w+)', normalized_no_trailing_semicolon))
    tables_in_query = {t for pair in tables_in_query for t in pair if t}
    if not tables_in_query.issubset(ALLOWED_TABLES):
        invalid = tables_in_query - ALLOWED_TABLES
        return False, f"Câu lệnh tham chiếu bảng không được phép: {invalid}"

    return True, "OK"

def run_sql_query(sql_query):
    is_safe, reason = is_sql_safe(sql_query)
    if not is_safe:
        return f"TỪ CHỐI THỰC THI: {reason}"
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return "\n".join(str(row) for row in rows)
    except Exception as e:
        return f"LỖI khi chạy SQL: {e}"
def agentic_answer(question):
    tool_choice = choose_tool_with_jev(question)

    if tool_choice == "semantic_search":
        tool_result = semantic_search(question)
    elif tool_choice == "run_sql_query":
        sql_query = generate_sql_with_ollama(question)
        print(f"[SQL sinh ra]: {sql_query}")
        tool_result = run_sql_query(sql_query)
        print(f"[Kết quả tool trả về]:\n{tool_result}\n")

        # Nếu bị từ chối hoặc lỗi, trả thẳng, không đưa cho model "tổng hợp" (tránh hallucination che giấu lỗi)
        if tool_result.startswith("TỪ CHỐI THỰC THI") or tool_result.startswith("LỖI"):
            return f"Xin lỗi, không thể lấy dữ liệu: {tool_result}"
    else:
        return f"Tool không xác định: {tool_choice}"

    final_prompt = f"""Câu hỏi của người dùng: {question}
        Dữ liệu trả về từ truy vấn (đây chính là kết quả đã lọc đúng theo câu hỏi):
        {tool_result}
        Hãy trả lời câu hỏi dựa trên dữ liệu trên."""

    final_response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": final_prompt}])
    return final_response["message"]["content"]


if __name__ == "__main__":
    question = "Liệt kê tất cả bài hát thuộc album Use Your Illusion I"
    print(f"\n--- Câu trả lời ---\n{agentic_answer(question)}")