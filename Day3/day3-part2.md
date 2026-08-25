# Handoff — RAG Eval Harness (pick up here next session)

## Where I am
Day 3–5 of the plan, essentially done. I have:
- A working RAG pipeline (FastAPI + ChromaDB + OpenAI) over `regulation.txt` (Migration Regulations).
- Chunks now tagged with a **regulation number** in metadata (`reg: "1.03"`, etc.) via a regex — this was the big unblock. Metadata now speaks the same language as my eval CSV.
- A 50-question eval CSV (`rag_eval_migration_regulations_50q.csv`), columns: `id, category, difficulty, question, ground_truth, source_reference`.
- An eval harness that loads the CSV, runs retrieval per question, and prints a recall score.

## The number I got
**36% (18/50).** Do NOT read this as "retrieval is broken." It's a measurement with two known issues dragging it down (below). The real retrieval quality is probably higher. Fix the measurement first, THEN trust the number, THEN tune.

## The exact next step (≈20 min)
Fix how the harness counts, then re-run:

1. **Abstention questions are being counted in recall — they shouldn't be.**
   - Denominator is 50; it should be ~43. The 7 abstention questions (`source_reference` = "Not in document…", ids around Q47–Q50 + any others) have no correct chunk to retrieve, so they can't have a recall score.
   - Confirm the harness actually skips them: `if base_reg(row["source_reference"]) is None: continue`. If `base_reg` returns None for those rows, they skip. Verify it's firing.
   - Score abstention SEPARATELY: for those 7, the pass condition is "did the system abstain / say I don't know?" — not retrieval.

2. **Verify the reg-number match on both sides (likely the real culprit).**
   - CSV ground truth is sub-clause level: `Reg 1.11(1)(c)`, `Reg 5.38(2)`. Chunk tags are base level: `1.11`, `5.38`.
   - `base_reg()` strips the CSV value to `1.11` — good. But confirm the CHUNK tag is also bare `1.11` (not `"Reg 1.11"` or `"1.11 "` with a space). A format mismatch = correct retrievals counted as misses.
   - Print `expected` vs `retrieved_regs` on 5 misses and eyeball whether any SHOULD have matched.

3. **Check the `unknown` rate.**
   - Print how many chunks are tagged `reg: "unknown"` vs a real number. If lots are `unknown`, naive 300-word chunking is splitting regulation numbers away from their text → tagging fails → false misses.
   - If `unknown` is high, that's a chunking fix (split on reg boundaries), not a retrieval-quality problem.

## How to read the result after fixing
- Low recall with the RIGHT reg genuinely absent from top-k → real retrieval miss → Day 6 tuning (chunk size, top-k, structure-aware chunking).
- Right reg present but counted as miss → matching bug (format/normalization).
- Cluster the misses by `difficulty` / `category` — if hard multi-section (Q33–Q41) fail most, that's the tuning roadmap.

## Then (Day 5 finish → Day 6)
- Add answer-quality scoring: LLM-as-judge for correctness + faithfulness (judge with a model STRONGER than gpt-4o-mini).
- Score abstention pass-rate separately (restraint, not recall).
- Day 6 = tune the Day 4 levers AGAINST the recall score, one variable at a time.

## Loose threads (parked, not urgent)
- `add` → `upsert` in `index()` (may already be done).
- Clean-ups: rename `find_regulations` → `find_reg`; drop the `context` field from `RegulationResponse`; fix leftover "EmailAnalysis" error string; add `encoding="utf-8"` to the file open; remove duplicate `import os`.
- Git commit agent (v1 works; v2 agent-loop pending).
- Langfuse observability (Day 6 half 2 — independent, can do anytime).
- Ruff linter configured; consider pre-commit hook later.
- READMEs written through Day 3; Day 4/5 still to write up.

## One-line reminder
The 36% is the measurement starting to tell the truth, not the system failing. Fix the count, trust the number, then tune.