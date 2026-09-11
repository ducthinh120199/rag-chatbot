from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

data = {
    "question": ["Referral Bonus là gì?"],
    "contexts": [["Nhân viên giới thiệu thành công 1 ứng viên trúng tuyển được thưởng 3,000,000 VNĐ..."]],
    "answer": ["Referral Bonus là khoản thưởng 3,000,000 VNĐ..."],
    "ground_truth": ["Referral Bonus là 3,000,000 VNĐ, chia 2 đợt: 1.5 triệu khi ký hợp đồng, 1.5 triệu sau 3 tháng thử việc."]
}

dataset = Dataset.from_dict(data)

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision], llm=llm, embeddings=embeddings)
print(result)