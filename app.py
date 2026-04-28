import streamlit as st
# from rag_core import retrieve_context, ask_groq
from data_chunk import generate_response, ask_llm
from dotenv import load_dotenv
load_dotenv()
st.set_page_config(page_title="News RAG", layout="wide")

st.title("News RAG – Ask your articles")

question = st.text_input("Ask a question about the news:")

if question:
    with st.spinner("Searching articles..."):
        contexts, sources = generate_response(question, k=5)

    with st.spinner("Thinking..."):
        answer = ask_llm(question, contexts)

    st.subheader("Answer")
    st.write(answer)

    st.subheader(" Sources")
    for src in sources:
        st.markdown(f"- {src}")

    with st.expander(" Retrieved Chunks"):
        for c in contexts:
            st.write(c)
            st.markdown("---")
