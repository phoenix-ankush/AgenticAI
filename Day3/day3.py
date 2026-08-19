from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

import chromadb
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI()
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)  # Replace with your actual OpenAI API key or ensure it's set in your environment variables.
chroma = chromadb.PersistentClient(
    path="./chroma_db"
)  # Ensure the path is correct and accessible.
collection = chroma.get_or_create_collection(
    "regulations"
)  # Ensure the collection name is correct.
BASE_DIR = Path(__file__).parent

SYSTEM_PROMPT = """You are a helpful assistant that provides information about regulations.
                You will be given a question and some context. Use the context to answer the question accurately. 
                If the context does not contain the answer, respond with "I don't know."
                Do not make up information. Be concise and clear in your response."""


class Query(BaseModel):
    query: str


class RegulationResponse(BaseModel):
    response: str
    context: str


def chunk(text: str, chunk_size: int = 300, overlap: int = 50):
    """
    Splits the input text into chunks of specified size with optional overlap.
    """
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i : i + chunk_size]))
        i += chunk_size - overlap
    return chunks


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
    return [item.embedding for item in resp.data]


def index(chunks: list[str], source: str) -> None:
    """
    Indexes the provided chunks into the ChromaDB collection with embeddings.
    """

    collection.add(
        documents=chunks,
        metadatas=[{"source": source}] * len(chunks),
        embeddings=embed_batch(chunks),
        ids=[f"{source}_{i}" for i in range(len(chunks))],
    )


def retrieve(question: str, k: int = 5):
    """
    Retrieves the top-k relevant documents from the ChromaDB collection based on the question.
    """
    question_embedding = embed_batch([question])[0]
    results = collection.query(query_embeddings=[question_embedding], n_results=k)
    return results["documents"][0], results["metadatas"][
        0
    ]  # Assuming we want the first set of documents and their metadata


@app.post("/index_regulations")
def index_regulations():
    """
    Endpoint to index regulations from a text file into the ChromaDB collection.
    """
    import os

    print("WORKING DIR:", os.getcwd())
    with open(BASE_DIR / "regulation.txt", "r") as f:
        text = f.read()

    chunks = chunk(text)
    index(chunks, source="regulation.txt")
    return {"message": "Regulations indexed successfully."}


@app.post("/query_regulations")
def query_regulations(query: Query):
    """
    Endpoint to query regulations based on the provided question.
    """
    queryContext, metadata = retrieve(query.query)
    print("Metadata", metadata)
    print("RETRIEVED:", queryContext)
    print("Collections:", collection.count())
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{queryContext}\n\nQuestion: {query.query}",
            },
        ],
        response_format=RegulationResponse,
    )
    result = response.choices[0].message.parsed
    if result is None:
        raise ValueError("Failed to parse the response into EmailAnalysis format.")
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
