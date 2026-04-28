# financial-news-rag
A Retrieval-Augmented Generation (RAG) application that answers questions over financial news articles using ChromaDB, OpenRouter LLMs, and Streamlit.

## Tech Stack
- ChromaDB (vector database)
- OpenRouter (LLM provider)
- Streamlit (UI)
- Python

## How it works
1. News articles are chunked and embedded
2. ChromaDB retrieves relevant chunks
3. An LLM answers questions using retrieved context only

## Setup

### 1. Clone repo
```bash
git clone https://github.com/your-username/news-rag-stream
