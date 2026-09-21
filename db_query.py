import ollama
import chromadb
from gemini_client import client

def embed_question_local(text):
    return ollama.embed(model="nomic-embed-text", input=text)["embeddings"][0]

chroma_client = chromadb.PersistentClient(path="./chroma_db_db")
collection = chroma_client.get_or_create_collection(name="chinook_tracks")

question = "Bài hát nào thuộc album Use Your Illusion I?"
question_vector = embed_question_local(question)

results = collection.query(
    query_embeddings=[question_vector],
    n_results=20,
    include=["documents", "distances", "metadatas"]
)

for i, doc in enumerate(results["documents"][0]):
    print(f"(distance: {results['distances'][0][i]:.4f}) {doc}")

context = "\n\n".join(results["documents"][0])

prompt = f"""Bạn là trợ lý trả lời câu hỏi dựa trên dữ liệu bài hát.
Chỉ trả lời dựa trên nội dung trong phần CONTEXT dưới đây.
Nếu CONTEXT không chứa thông tin liên quan, hãy trả lời: "Tôi không tìm thấy thông tin này."

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
print(f"\n--- Câu trả lời ---\n{response.text}")