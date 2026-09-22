import streamlit as st
from agentic_query import agentic_answer

st.set_page_config(page_title="Chinook Music Assistant", page_icon="🎵")
st.title("🎵 Chinook Music Assistant")
st.caption("Agentic RAG: Jev chọn tool (semantic search / SQL), Ollama sinh câu trả lời")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Hỏi gì đó về nhạc, album, thể loại..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Đang xử lý..."):
            answer = agentic_answer(question)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})