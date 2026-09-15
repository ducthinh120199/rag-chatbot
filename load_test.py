import requests
import concurrent.futures

URL = "http://127.0.0.1:49810/api/chat"  # thay đúng port từ minikube service

def send_request(i):
    response = requests.post(URL, json={
        "model": "llama3.2",
        "messages": [{"role": "user", "content": f"Viết 1 đoạn văn dài 200 từ về chủ đề số {i}"}],
        "stream": False
    })
    print(response.status_code, response.text)

with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    executor.map(send_request, range(50))