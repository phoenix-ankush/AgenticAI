# Day 3 — Full RAG Pipeline in One Sitting

**Goal:** Ask a question and get an answer grounded in my *own* documents, with an "I don't know" fallback — not the model's training data.

## What I built
A complete Retrieval-Augmented Generation pipeline over a local document:
- **Ingest** — read a `.txt` file, chunk it into overlapping word-windows
- **Embed + store** — batch-embed chunks with `text-embedding-3-large`, store in a persistent ChromaDB collection
- **Retrieve** — embed the question, pull the top-k nearest chunks by similarity
- **Generate** — feed retrieved chunks to `gpt-4o-mini` and answer *from that context*, returning "I don't know" when the docs don't cover it

Two endpoints: `POST /index_regulations` (one-time ingest) and `POST /query_regulations` (retrieve → generate).

## What I learned
- **RAG is retrieve-then-generate.** The LLM never saw my documents and can't fit the whole corpus in context. So I retrieve the relevant pieces and hand them to the model in the prompt. The model doesn't *know* the answer — it's given the material to answer from.
- **Embeddings turn meaning into geometry.** Text becomes a vector; semantically similar text lands close together. Retrieval is nearest-neighbour search in that space, measured by cosine (angle, not magnitude — so length differences don't distort the meaning comparison).
- **The symmetry:** embed chunks once at ingest, embed the question at query time *with the same model*, compare. The same-model rule is load-bearing — mismatched models put query and stored vectors in different spaces and retrieval returns garbage.
- **Ingest is a one-time step, not per-request.** Populate the index once; the query endpoint just reads from it.
- **Retrieval quality is the ceiling.** If the right chunk isn't retrieved, no prompt or model cleverness recovers it — the model can only answer from what it's handed.

## What broke and how I fixed it
> The richest debugging day so far — the bugs were spread across every stage.
- **Empty responses / "I don't know" on valid questions.** Diagnosed by reading the API output: `prompt_tokens=134` was the smoking gun — far too small to contain retrieved context, proving the context never reached the prompt. Lesson: **when a RAG system says "I don't know," suspect the context, not the model** — it's usually faithfully reporting that it was handed nothing.
- **`RETRIEVED: []` → empty collection.** Traced the empty context back with `collection.count()`. The collection had never been populated because ingest hadn't succeeded this session.
- **`FileNotFoundError: 'regulation.txt'`** even though the file sat next to the script. Cause: relative paths resolve from the **current working directory** (where Python was launched), not the script's location. I'd launched from a different folder. Fix: anchor the path with `Path(__file__).parent / "regulation.txt"` so it's location-independent.
- **Context injected in the wrong role.** I first passed retrieved chunks as an `assistant` message — telling the model "you already said this," which is false and breaks the grounding contract. Retrieved context is reference material and belongs in the **user** message, joined into a single string. Getting the role/placement right *is* the RAG technique; everything else just exists to get the right text into that message.
- **Endpoint returned `null`** — ended on `print(response)` instead of returning `.choices[0].message.parsed`.

## Key debugging habit
**Instrument each stage's output so a failure names *which* stage broke.** Printing chunk count, `collection.count()` after indexing, and the retrieved chunks before the LLM call turned a vague "no response" into a precise trace: prompt → retrieval → collection → ingest → file path.

## Key code
```python
BASE_DIR = Path(__file__).parent
text = (BASE_DIR / "regulation.txt").read_text(encoding="utf-8")

def retrieve(question, k=5):
    q_emb = embed_batch([question])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0], res["metadatas"][0]

# context goes in the USER message, joined to one string
context_block = "\n\n".join(retrieved_chunks)
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"},
]
```

## What's next
**Day 4 — Make retrieval good.** Deliberately break the naive 300-word chunking, then fix it: chunk size/overlap, structure-aware splitting, top-k tuning, metadata filtering. Less API, more judgment — the part that separates real RAG from a copied tutorial. It also makes me *want* evals (Day 5), because eyeballing "did that get better?" stops scaling.