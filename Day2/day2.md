# Day 2 — Structured JSON Output with Pydantic

**Goal:** Make the LLM return validated, schema-conforming JSON my code can trust — not free-form text I have to parse.

## What I built
A `POST /analyze_email` endpoint that takes raw, unstructured email text and returns a structured `EmailAnalysis` object (sender intent, urgency, action items, summary) as clean JSON.

## What I learned
- **Structured Outputs vs. "please return JSON."** Prompting a model to "return JSON" is fragile — it may wrap output in markdown fences, add a chatty preamble, or drift from the shape I wanted. Structured Outputs *guarantees* the response conforms to the schema, because generation is constrained at decode time — not asked politely.
- **Pydantic is a general skill, not an AI one.** Defining the expected shape as a typed `BaseModel` gives validation for free. Same library powers FastAPI request bodies, config, data pipelines — the LLM is just one more place it's useful.
- **The architectural shift:** with a schema enforced, the LLM stops being a text generator I read and becomes a **typed component in a system** whose output downstream code can depend on. This is the foundation for tool calls and agents later (same mechanism underneath).
- **`.parse()` returns a typed object, not a string.** `completion.choices[0].message.parsed` is a fully instantiated Pydantic model — dot access, already-typed fields, no `json.loads`, no regex, no cleanup.

## What broke and how I fixed it
> The code "worked" but had a conceptual bug — the most useful kind to catch.
- **Wrapped a non-streaming call in `StreamingResponse`.** I copied Day 1's stream→yield pattern into the parse endpoint. But `.parse()` is *not* a stream — it returns one complete, validated object all at once (it has to see the whole output to validate it against the schema). Streaming and parsing are opposed by definition. Fix: **delete** the streaming and just `return` the Pydantic object — FastAPI serializes it to JSON natively. The tell was naming the variable `stream` when it wasn't one.
- **`result.json()` is deprecated (Pydantic v2)** — it's `model_dump_json()` now. But returning the object directly means I don't need it at all.
- **Didn't handle refusal.** `.parsed` can be `None` if the model refuses. Added an `if result is None` guard reading `.refusal` — small production-discipline habit.
- **Redundant / leftover code:** passing `api_key=os.getenv(...)` is unnecessary (the SDK reads the env var itself); `camelCase` field names → `snake_case`; and the function was still named `ask_question` from Day 1.

## Key concept locked in
**Know whether you have the whole thing or a piece of it.**
- Day 1: pieces arriving over time → `stream` + `yield`.
- Day 2: the whole validated object at once → `parse` + `return`.

Matching the response mechanism to the situation is the actual skill; the SDK method names are just labels on it.

## Key code
```python
class EmailAnalysis(BaseModel):
    sender_intent: str
    urgency: str
    action_items: list[str]
    summary: str


@app.post("/analyze_email")
def analyze_email(query: Query) -> EmailAnalysis:
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract structured info from the email."},
            {"role": "user", "content": query.email_text},
        ],
        response_format=EmailAnalysis,
    )
    result = completion.choices[0].message.parsed
    if result is None:
        raise ValueError(completion.choices[0].message.refusal)
    return result
```

## What's next
**Day 3 — Full RAG in one sitting:** ingest docs, chunk, embed, store in a local vector DB (Chroma), and answer questions *from my own documents* with citations. Level 2 — real system-building, not API calls.