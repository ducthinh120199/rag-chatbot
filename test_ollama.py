import ollama

response = ollama.chat(model="llama3.2", messages=[
    {"role": "user", "content": "Xin chào, bạn là ai?"}
])

print(response["message"]["content"])