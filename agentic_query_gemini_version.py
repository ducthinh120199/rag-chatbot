"""
BẢN CŨ — dùng Gemini function calling để chọn tool (thay vì Jev).
Giữ lại để so sánh với agentic_query.py (bản mới, dùng Jev).
Khác biệt duy nhất nằm ở agentic_answer() — phần còn lại giống hệt bản mới.
"""

from gemini_client import client
import psycopg2
import ollama
import chromadb
import re

ALLOWED_TABLES = {"track", "album", "genre"}

SCHEMA_INFO = """
Bảng track(track_id, name, composer, album_id, genre_id, unit_price, milliseconds)
Bảng album(album_id, title, artist_id)
Bảng genre(genre_id, name)

QUAN TRỌNG: Đây là PostgreSQL. Dùng dấu nháy ĐƠN '...' cho giá trị chuỗi (ví dụ: WHERE title = 'ABC').
KHÔNG dùng dấu nháy kép "..." cho giá trị chuỗi — dấu nháy kép chỉ dùng cho tên cột/bảng.
"""

# --- Định nghĩa tool cho Gemini function calling (Jev KHÔNG cần cấu trúc này) ---
tools = [
    {
        "name": "semantic_search",
        "description": "Tìm kiếm ngữ nghĩa khi câu hỏi mang tính mô tả, không cần liệt kê chính xác toàn bộ (ví dụ: 'bài hát nào giống thể loại Rock, năng lượng cao')",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    },
    {
        "name": "run_sql_query",
        "description": "Chạy câu SQL khi câu hỏi cần liệt kê TOÀN BỘ, ĐẾM CHÍNH XÁC, hoặc lọc theo điều kiện rõ ràng (ví dụ: 'liệt kê tất cả bài hát thuộc album X', 'có bao nhiêu bài hát thể loại Rock')",
        "parameters": {
            "type": "object",
            "properties": {"sql_query": {"type": "string"}},
            "required": ["sql_query"]
        }
    }
]

def semantic_search(query):
    vector = ollama.embed(model="nomic-embed-text", input=query)["embeddings"][0]
    chroma_client = chromadb.PersistentClient(path="./chroma_db_db")
    collection = chroma_client.get_or_create_collection(name="chinook_tracks")
    results = collection.query(query_embeddings=[vector], n_results=5, include=["documents"])
    return "\n".join(results["documents"][0])

def is_sql_safe(sql_query):
    normalized = sql_query.strip().lower()
    if not normalized.startswith("select"):
        return False, "Chỉ cho phép câu lệnh SELECT."
    dangerous_keywords = ["insert", "update", "delete", "drop", "alter", "truncate", "create", "grant", "--", ";"]
    for keyword in dangerous_keywords:
        if keyword in normalized:
            return False, f"Câu lệnh chứa từ khóa không được phép: '{keyword}'"
    tables_in_query = set(re.findall(r'\bfrom\s+(\w+)|\bjoin\s+(\w+)', normalized))
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
        conn = psycopg2.connect(host="localhost", port=5432, dbname="chinook", user="postgres", password="pass")
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return "\n".join(str(row) for row in rows)
    except Exception as e:
        return f"LỖI khi chạy SQL: {e}"

# --- Bộ não điều phối — BẢN CŨ: 1 lần gọi Gemini vừa CHỌN TOOL vừa (nếu SQL) tự SINH CÂU SQL luôn ---
def agentic_answer(question):
    prompt = f"Schema database:\n{SCHEMA_INFO}\n\nCâu hỏi: {question}"

    # Đây là lần gọi generate_content TỐN QUOTA — chính là nguyên nhân bị 429 hôm nay
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={"tools": [{"function_declarations": tools}]}
    )

    part = response.candidates[0].content.parts[0]

    if hasattr(part, "function_call") and part.function_call:
        fn_name = part.function_call.name
        fn_args = part.function_call.args

        print(f"[Model chọn tool: {fn_name}, args: {dict(fn_args)}]")

        if fn_name == "semantic_search":
            tool_result = semantic_search(fn_args["query"])
        elif fn_name == "run_sql_query":
            tool_result = run_sql_query(fn_args["sql_query"])
            print(f"[Kết quả tool trả về]:\n{tool_result}\n")
        else:
            return f"Tool không xác định: {fn_name}"

        final_prompt = f"""Câu hỏi của người dùng: {question}
Dữ liệu trả về từ truy vấn (đây chính là kết quả đã lọc đúng theo câu hỏi):
{tool_result}
Hãy trả lời câu hỏi dựa trên dữ liệu trên."""

        # Lưu ý: bản gốc gửi trước đó có lỗi final_response.text (sai cú pháp cho ollama.chat) — đã sửa lại đúng ở đây
        final_response = ollama.chat(model="llama3.2", messages=[{"role": "user", "content": final_prompt}])
        return final_response["message"]["content"]
    else:
        return part.text


if __name__ == "__main__":
    question = "Liệt kê tất cả bài hát thuộc album Use Your Illusion I"
    print(f"\n--- Câu trả lời (bản Gemini function calling) ---\n{agentic_answer(question)}")