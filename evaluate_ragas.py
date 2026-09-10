from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=os.getenv("GEMINI_API_KEY"))
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=os.getenv("GEMINI_API_KEY"))

data = {
    "question": ["Referral Bonus là gì?"],
    "contexts": [["Nhân viên giới thiệu thành công 1 ứng viên trúng tuyển được thưởng 3,000,000 VNĐ, chia làm 2 đợt..."]],
    "answer": ["Referral Bonus là khoản thưởng 3,000,000 VNĐ cho nhân viên giới thiệu ứng viên trúng tuyển, chia làm 2 đợt."]
}

dataset = Dataset.from_dict(data)

result = evaluate(dataset, metrics=[faithfulness, answer_relevancy], llm=llm, embeddings=embeddings)
print(result)