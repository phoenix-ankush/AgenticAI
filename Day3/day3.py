from email.mime import text
from importlib.metadata import metadata
from itertools import count

from dotenv import load_dotenv

load_dotenv()

import os
from pathlib import Path

import chromadb
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import re
import csv

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
REG_PATTERN = re.compile(r'\b(\d+\.\d+)\b')   # Pattern to match regulation numbers like 1.1, 2.3, etc.")


SYSTEM_PROMPT = """You are a helpful assistant that provides information about regulations.
                You will be given a question and some context. Use the context to answer the question accurately. 
                If the context does not contain the answer, respond with "I don't know."
                Do not make up information. Be concise and clear in your response."""
SEPARATOR = "1994 No. 268 MIGRATION REGULATIONS"

NAME_PATTERN = re.compile(r'^\s*-\s*(.+)')   # capture what's after the leading "- "

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

def find_regulations(text: str) -> str | None:
    """
    Extracts regulation numbers from the provided text using a regex pattern.
    Returns a string of found regulations or None if no regulations are found.
    """
    matches = REG_PATTERN.search(text)
    if matches:
        return matches.group(1)
    return None


def embed_batch(texts: list[str]) -> list[list[float]]:
    resp = client.embeddings.create(model="text-embedding-3-large", input=texts)
    return [item.embedding for item in resp.data]


def index(chunks: list[str], tags: list[str]) -> None:
    """
    Indexes the provided chunks into the ChromaDB collection with embeddings.
    """

    collection.upsert(
        documents=chunks,
        metadatas=[{"reg": tag} for tag in tags],
        embeddings=embed_batch(chunks),
        ids=[f"{tag}_{i}" for i, tag in enumerate(tags)],
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


def base_reg(ref: str) -> str:
    m = re.search(r"(\d+\.\d+)", ref)
    if m:
        return m.group(1)
    return ref

def run_evaluation(csv_path, k: int = 5):
    hits, total = 0, 0
    missed = []
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            expected = base_reg(row["source_reference"])
            if "not in document" in row["source_reference"].lower():
                continue   # abstention question — score separately, not in recall
            if expected == "None":
                continue
            _, metadata = retrieve(row["question"], k=k)
            retrieved_regs = [base_reg(m["reg"]) for m in metadata]
            total += 1
            if expected in retrieved_regs:
                hits += 1
            else:
                missed.append({
                    "question": row["question"],
                    "expected": expected,
                    "retrieved": retrieved_regs,
                })
    return {
        "accuracy": round(hits / total, 4) if total else 0,
        "hits": hits,
        "total": total,
        "missed": missed,
    }
    

def indexing_based_one_reg(text: str, max_words: int = 500):
    """
    Indexes regulations from a text file into the ChromaDB collection.
    """
    
    sections = text.split(SEPARATOR)
    chunks, tags = [], []

    for section in sections:
        name = find_chunk_name(section)
        if name is None:  # no name found → skip this fragment
            continue
        m = re.search(r'(\d+\.\d+)', name)
        tag = m.group(1) if m else name.strip()

        words = section.split()
        if len(words) <= max_words:
            chunks.append(section)
            tags.append(tag)
        else:
            # Split the section into smaller chunks
            for i in range(0, len(words), max_words):
                chunk = " ".join(words[i:i + max_words])
                chunks.append(chunk)
                tags.append(tag)
    return chunks,tags
    

def find_chunk_name(text: str) -> str | None:
    m = NAME_PATTERN.search(text)
    return m.group(1).strip() if m else None    

@app.post("/index_regulations")
def index_regulations():
    """
    Endpoint to index regulations from a text file into the ChromaDB collection.
    """
    import os

    print("WORKING DIR:", os.getcwd())
    with open(BASE_DIR / "regulation.txt", "r") as f:
        text = f.read()

    chunks, tags = indexing_based_one_reg(text)
    index(chunks, tags=tags)
    return {"message": "Regulations indexed successfully."}


@app.post("/query_regulations")
def query_regulations(query: Query):
    """
    Endpoint to query regulations based on the provided question.
    """
    queryContext, metadata = retrieve(query.query)
    for m in metadata:
        print("Metadata:", m)

    # print("Metadata", metadata)
    # print("RETRIEVED:", queryContext)
    # print("Collections:", collection.count())
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

    retrieved_regs = [m["reg"] for m in metadata]
    expected = "1.09"                       # from your CSV's source_reference
    hit = expected in retrieved_regs
    print("HIT" if hit else "MISS", "| retrieved:", retrieved_regs)

    if result is None:
        raise ValueError("Failed to parse the response into EmailAnalysis format.")
    return result



@app.post("/run_evaluation")
def evaluate(k: int = 5):
    csv_path = BASE_DIR / "evals_questions.csv"
    return run_evaluation(csv_path, k=k)
# indexing_based_one_reg()

@app.post("/reset")
def reset():
    chroma.delete_collection("regulations")
    global collection
    collection = chroma.get_or_create_collection("regulations")
    return {"message": "Collection reset — now empty. Re-index next."}

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
