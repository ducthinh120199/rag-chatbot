from google import genai
import os
from dotenv import load_dotenv
import chromadb
from gemini_client import client, embed_text
import logging
logging.getLogger("google_genai").setLevel(logging.ERROR)

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_handbook")

question = "Cho tôi biết các thông tin liên quan về referral bonus?"
question_vector = embed_text(question)

results = collection.query(
    query_embeddings=[question_vector],
    n_results=3,
    include=["documents", "distances", "metadatas"]
)

for i, doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]
    print(f"[Nguồn: {source}] {doc[:100]}...")

context = "\n\n".join(results["documents"][0])

prompt = f"""Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu công ty.
Chỉ trả lời dựa trên nội dung trong phần CONTEXT dưới đây.
Nếu CONTEXT không chứa thông tin liên quan, hãy trả lời: "Tôi không tìm thấy thông tin này trong tài liệu."

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)

print(response.text)