from openai import OpenAI
import os
import chromadb
from dotenv import load_dotenv
import pandas as pd
import json
from pprint import pprint

load_dotenv()

client=OpenAI(base_url="https://openrouter.ai/api/v1",
              api_key=os.getenv("router_api"))


df=pd.read_csv("financial_news_articles.csv")
assert "article_text" in df.columns
assert "url" in df.columns
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
    separators=["\n\n","\n","."," ",""]
    )


chunks=[]
for i,row in df.iterrows():
    # print(f"Splitting article {i+1} with url {row['url']}")
    article_text=row["article_text"]
    url=row["url"]

    # split the articles into chunks
    article_chunks=text_splitter.split_text(article_text)

    for chunk_idx, chunk in enumerate(article_chunks):
        chunks.append({
            "id": f"{i}_{chunk_idx}",
            "text": chunk,
            "metadata": {
                "url": url,
                "article_index": i,
                "chunk_index": chunk_idx
            }
        })



client=chromadb.PersistentClient(path="./chroma_db")

collection=client.get_or_create_collection(
    name="financial_news_articles",
    # metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=[chunk["id"] for chunk in chunks],
    documents=[chunk["text"] for chunk in chunks],
    metadatas=[chunk["metadata"] for chunk in chunks]
)

def generate_response(query, k=5):
    results = collection.query(
        query_texts=[query],
        n_results=k,
        include=["documents", "metadatas"]
    )

    contexts = results["documents"][0]
    sources = [m["url"] for m in results["metadatas"][0]]

    return contexts, list(set(sources))

# def generate_response(query, k=5):
#     results = collection.query(
#         query_texts=[query],
#         n_results=k,
#         include=["documents", "metadatas"]
#     )

#     retrieved_chunks = results["documents"][0]
#     metadatas = results["metadatas"][0]

#     sources = [meta["url"] for meta in metadatas]

#     debug_text = f"Query: {query}\n\nRetrieved Chunks:\n"
#     for i, (chunk, source) in enumerate(zip(retrieved_chunks, sources)):
#         debug_text += f"{i+1}. {chunk}\n(Source: {source})\n\n"

#     return retrieved_chunks, sources, debug_text


def ask_llm(question, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a very helpful assistant.

Answer the question ONLY using the context below.
If the answer is not present, say:
"I don't know based on the provided articles."

Context:
{context_text}

Question:
{question}

Answer:
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-3-70b-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content

