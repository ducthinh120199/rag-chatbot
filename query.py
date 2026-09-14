import chromadb
from gemini_client import client, embed_text
from rank_bm25 import BM25Okapi
from chunk_utils import get_chunk_id
from sentence_transformers import CrossEncoder
import time
import ollama

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="company_handbook")

question = "Referral Bonus là gì?"

# --- Chuẩn bị dữ liệu cho BM25 (cần toàn bộ chunk, không phải top-k) ---
all_data = collection.get(include=["documents", "metadatas"])
all_chunks = all_data["documents"]
all_sources = all_data["metadatas"]
all_ids = all_data["ids"]

tokenized_chunks = [chunk.lower().split() for chunk in all_chunks]
bm25 = BM25Okapi(tokenized_chunks)

# --- Dense search (đã có từ trước) ---
question_vector = embed_text(question)

results = collection.query(
    query_embeddings=[question_vector],
    n_results=15,
    include=["documents", "distances", "metadatas"]
)

print("--- Dense search (theo ý nghĩa) ---")
for i, doc in enumerate(results["documents"][0]):
    source = results["metadatas"][0][i]["source"]
    distance = results["distances"][0][i]
    print(f"[{source}] (distance: {distance:.4f}) {doc[:80]}...")

# --- Sparse search / BM25 (mới thêm) ---
tokenized_question = question.lower().split()
bm25_scores = bm25.get_scores(tokenized_question)

bm25_ranking = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:15]

print("\n--- Sparse search / BM25 (theo từ khóa) ---")
for rank, idx in enumerate(bm25_ranking):
    source = all_sources[idx]["source"]
    print(f"[{source}] (score: {bm25_scores[idx]:.2f}) {all_chunks[idx][:80]}...")

def reciprocal_rank_fusion(dense_ids, sparse_ids, k=60):
    scores = {}
    
    for rank, doc_id in enumerate(dense_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    for rank, doc_id in enumerate(sparse_ids):
        scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

#--- Reciprocal Rank Fusion (RRF) ---
dense_ids = [get_chunk_id(doc) for doc in results["documents"][0]]
sparse_ids = [get_chunk_id(all_chunks[idx]) for idx in bm25_ranking]

fused = reciprocal_rank_fusion(dense_ids, sparse_ids)
print("\n--- Kết quả sau RRF ---")
for doc_id, score in fused[:3]:
    print(f"{doc_id[:8]} (RRF score: {score:.4f})")

def get_text_by_id(chunk_id, all_chunks, all_ids):
    idx = all_ids.index(chunk_id)
    return all_chunks[idx]

candidate_ids = [doc_id for doc_id, score in fused[:15]]
candidate_chunks = [get_text_by_id(cid, all_chunks, all_ids) for cid in candidate_ids]

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [(question, chunk) for chunk in candidate_chunks]
rerank_scores = reranker.predict(pairs)

reranked = sorted(zip(candidate_chunks, rerank_scores), key=lambda x: x[1], reverse=True)

print("\n--- Kết quả sau Reranking ---")
for chunk, score in reranked[:3]:
    print(f"(score: {score:.4f}) {chunk[:80]}...")

final_chunks = [chunk for chunk, score in reranked[:3]]
context = "\n\n".join(final_chunks)

prompt = f"""Bạn là trợ lý trả lời câu hỏi dựa trên tài liệu công ty.
Chỉ trả lời dựa trên nội dung trong phần CONTEXT dưới đây.
Nếu CONTEXT không chứa thông tin liên quan, hãy trả lời: "Tôi không tìm thấy thông tin này trong tài liệu."

CONTEXT:
{context}

CÂU HỎI: {question}

TRẢ LỜI:"""

t0 = time.time()
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=prompt
)
t1 = time.time()
print(f"Generation (Gemini): {t1-t0:.3f}s")
print(f"\n--- Câu trả lời (Gemini) ---\n{response.text}")

t6 = time.time()
ollama_response = ollama.chat(model="llama3.2", messages=[
    {"role": "user", "content": prompt}
])
t7 = time.time()
print(f"Generation (Ollama): {t7-t6:.3f}s")
print(f"\n--- Câu trả lời (Ollama) ---\n{ollama_response['message']['content']}")

t4 = time.time()
mistral_response = ollama.chat(model="mistral", messages=[{"role": "user", "content": prompt}])
t5 = time.time()
print(f"\n--- Ollama Mistral ({t5-t4:.2f}s) ---\n{mistral_response['message']['content']}")