# Day 1 — Streaming FastAPI Endpoint over the OpenAI API

**Goal:** Turn a Python script into a deployable web service that streams an LLM response token-by-token.

## What I built
A FastAPI app with a `POST /ask` endpoint that takes a question and streams the model's answer back as it's generated (the "typewriter" effect), rather than waiting for the full response.

## What I learned
- **FastAPI POST endpoints** — defining a request body with a Pydantic model (`Query`) and handling a POST request.
- **How streaming works** — the OpenAI API can return the response incrementally as **server-sent events (SSE)**. Instead of one big response, you get a sequence of small chunks, each carrying a `delta` (the *new* piece of text since the last chunk). The full answer is the concatenation of all deltas, in order.
- **Python generators + `yield`** — a generator hands out one piece at a time and *pauses* until the next is requested. It streams each delta straight through to the caller instead of accumulating them. The "not saving" is the point — it's what makes the response appear progressively.
- **Server vs. client asymmetry** — the server yields and forgets; the client accumulates into a variable and re-renders the UI on each piece.
- **How the stream ends** — three nested signals: the semantic event (`response.completed` / `finish_reason: "stop"`), the SSE sentinel (`data: [DONE]`), and the transport (HTTP connection closes → client sees `done: true`).
- **Python environment hygiene** — `venv` for per-project isolation, `requirements.txt` as the rebuild recipe, `.env` for secrets, and `.gitignore` to keep secrets out of version control.

## What broke and how I fixed it
- **`AttributeError: 'ChoiceDelta' object has no attribute 'get'`** — my code used `chunk.choices[0].delta.get("content")`, which is the old (pre-1.0) OpenAI library's dict-style access. The current library returns typed objects, so the fix is dot access: `chunk.choices[0].delta.content`. Lesson: dict access (`.get()`) vs. attribute access (`.content`) — an "object has no attribute 'get'" error means it's an object, use a dot.
- **`pip install requirement.txt` failed** — missing the `-r` flag (pip thought the file was a package name) and a filename typo (`requirement` vs `requirements`). Correct: `pip install -r requirements.txt`.
- **Debugging setup** — learned to read a traceback **bottom-up** (last line = what broke, first line mentioning my own file = where), and to use `breakpoint()` / VS Code breakpoints. Key gotcha: VS Code breakpoints only hit when the app is **launched by the debugger (F5)**, not when started via `uvicorn` in the terminal; `--reload` also has to be off.
- **Git two-account conflict** — the machine had a corporate GitHub account cached in the Keychain, causing `repository not found` then `403 write access not granted` when pushing to my personal repo. Fixed by scoping the personal account to this repo only: local `user.name`/`user.email` (no `--global`) and a remote URL carrying the personal account, so the corporate setup stays untouched.

## Key commands
```bash
# environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# run
uvicorn Day1:app --reload

# test
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Say hello in one sentence."}'
```

## What's next
**Day 2 — structured JSON output:** a second endpoint that returns validated JSON via Pydantic, so the model's output is machine-readable — the foundation for tool calls and extraction later.