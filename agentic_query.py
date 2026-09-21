from gemini_client import client
import psycopg2
import ollama
import chromadb

# --- Định nghĩa các tool mà model có thể chọn gọi ---
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

# --- Tool 1: Semantic search qua ChromaDB ---
def semantic_search(query):
    vector = ollama.embed(model="nomic-embed-text", input=query)["embeddings"][0]
    chroma_client = chromadb.PersistentClient(path="./chroma_db_db")
    collection = chroma_client.get_or_create_collection(name="chinook_tracks")
    results = collection.query(query_embeddings=[vector], n_results=5, include=["documents"])
    return "\n".join(results["documents"][0])

# --- Tool 2: Chạy SQL trực tiếp trên Postgres ---
def run_sql_query(sql_query):
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

# --- Bộ não điều phối: model tự quyết định gọi tool nào ---
def agentic_answer(question):
    schema_info = """
    Bảng track(track_id, name, composer, album_id, genre_id, unit_price, milliseconds)
    Bảng album(album_id, title, artist_id)
    Bảng genre(genre_id, name)

    QUAN TRỌNG: Đây là PostgreSQL. Dùng dấu nháy ĐƠN '...' cho giá trị chuỗi (ví dụ: WHERE title = 'ABC'). 
    KHÔNG dùng dấu nháy kép "..." cho giá trị chuỗi — dấu nháy kép chỉ dùng cho tên cột/bảng.
    """

    prompt = f"Schema database:\n{schema_info}\n\nCâu hỏi: {question}"

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
        Hãy trả lời câu hỏi dựa trên dữ liệu trên. Toàn bộ dữ liệu này đã được lọc đúng theo điều kiện trong câu hỏi."""
        final_response = client.models.generate_content(model="gemini-2.5-flash", contents=final_prompt)
        return final_response.text
    else:
        return part.text


if __name__ == "__main__":
    question = "Liệt kê tất cả bài hát thuộc album Use Your Illusion I"
    print(f"\n--- Câu trả lời ---\n{agentic_answer(question)}")