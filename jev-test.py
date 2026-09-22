import requests
import os
from dotenv import load_dotenv

load_dotenv()

response = requests.post(
    "https://ai-gateway.vercel.sh/typesafe/v1/systemone",
    headers={
        "Authorization": f"Bearer {os.getenv('AI_GATEWAY_API_KEY')}",
        "Content-Type": "application/json"
    },
    json={
        "model": "typesafe-ai/jev",
        "state": "Liệt kê tất cả bài hát thuộc album Use Your Illusion I",
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
print(response.status_code)
print(response.json())