# Decision-model experiments (Laya)

A separate line of work from the vLLM engine. These are **experiments, not library
code**: standalone scripts that probe what a non-autoregressive System-One decision
model ([`laya`](https://pypi.org/project/laya/), ModernBERT-based) can and cannot
represent. Nothing here is imported by `llm_lab`, and nothing here is covered by the
test suite in `tests/`.

Two threads, run in order. Each script loads the model once, prints the full
probability distribution for every case, and keeps its configuration frozen so later
runs stay comparable.

## Thread 1 — support ticket triage

| Script | What it asks | Result |
|---|---|---|
| [`ticket_triage.py`](ticket_triage.py) | `choice` over 4 departments, 10 tickets, plus an abstain rule on confidence and margin | 7/9 correct; both errors and the ambiguous case fell below 0.50 while every correct answer sat above 0.75 |
| [`ticket_refund_intent.py`](ticket_refund_intent.py) | `noul` — "is the customer explicitly asking for a refund?", 12 cases | 8/12. All four hard negatives failed, and the classes are **not threshold-separable** |
| [`ticket_resolution_intent.py`](ticket_resolution_intent.py) | `choice` over 6 resolutions, 12 cases | 6/12. Perfect on explicit requests, 0/6 on everything else |

**Finding.** The refund head does not answer the question it was given. Money without a
complaint scores 0.041; a complaint without money scores 0.032; **both together score
0.68–0.79**. It estimates "is this a transaction grievance that might end in money going
back", not "did the customer ask". `"Can I get my money back?"` (0.725) ranks *below*
`"My order is wrong."` (0.793), so no cutoff fixes it. Widening the option set did not
help: `no_resolution_stated` became a residual bucket, winning on the two cases where
the customer *did* request help and losing where they requested nothing.

## Thread 2 — agent action guard

Can the model judge whether an agent's proposed tool call matches the user's request?
All four scripts share **the same 20 hand-labelled examples**, imported from
`guard1_alignment.py` rather than copied, so the comparison holds by construction.
Categories: A clearly aligned, B clearly misaligned, C same tool with opposite correct
answers, D underspecified.

| Script | Architecture | Clear | Ambiguous caught | Wrong |
|---|---|---|---|---|
| [`guard1_alignment.py`](guard1_alignment.py) | `aligned` / `not_aligned`, one question | **15/15** | 2/5 | **0** |
| [`guard2_sufficiency_gate.py`](guard2_sufficiency_gate.py) | sufficiency gate (request + action) → alignment | 8/15 | 5/5 | 0 |
| [`guard3_intent_gate.py`](guard3_intent_gate.py) | sufficiency gate (request only) → alignment | 1/15 | 3/5 | 0 |
| [`guard4_threeway.py`](guard4_threeway.py) | one three-way label space | 11/15 | 0/5 | 2 |

**Finding 1 — intent/action alignment works.** Category C is 5/5. On byte-identical
action text, `delete_file("report.csv")` swings 0.664 in probability and
`send_email(...)` swings 0.533, purely from rewording the request. The model compares
the two halves of the state; it has not simply learned that delete is bad.

**Finding 2 — "underspecified" is not representable here.** Three structurally
different attempts, three failures with different signatures:

- **Guard 2** — the sufficiency label matched the *alignment* ground truth 15/15. Every
  precise-but-misaligned request came back `insufficient`. The action was in the state,
  so the question was answered as alignment under another name.
- **Guard 3** — removing the action removed the only signal. 17 of 20 cases landed
  between 0.512 and 0.589, and the separation **inverted**: vague requests scored as
  *more* specific (mean 0.638) than precise ones (0.596).
- **Guard 4** — `insufficient_intent` was never predicted once in 20 examples. Mean
  probability 0.182 against a uniform 0.333, so it never exceeded chance; adding it also
  broke 3 of 8 previously-correct aligned cases, because the broad `not_aligned`
  criterion absorbed the mass and took 14 of 20 top slots.

The consistent shape across both threads: this model does **relational comparison
between things present in the input**, and cannot reason about what is *absent* from it.
The refund head showed the same failure on a propositional question.

**Finding 3 — what actually detects ambiguity is low confidence.** In every task the
ambiguous cases were caught by the confidence/margin thresholds, never by a semantic
label. Guard 2's perfect 5/5 capture was the threshold, not the head. Thresholds sit on
knife edges: guard 4 example 20 scored 0.596 against a 0.600 cutoff, and guard 1
examples 16 and 19 cleared a 0.65 cutoff at 0.659 and 0.660.

## Running them

`laya` is a declared dependency, so `uv sync` installs it. The first run downloads
ModernBERT-large (~1.5 GB) to the Hugging Face cache.

```bash
uv run python src/llm_lab/inference/decision/ticket_triage.py
uv run python src/llm_lab/inference/decision/guard1_alignment.py
uv run python src/llm_lab/inference/decision/guard4_threeway.py
```

Runs are deterministic — repeated runs reproduce identical probabilities.

Laya prints a `RuntimeWarning` on load saying the published checkpoint ships invalid
temperatures and that affected confidences are uncalibrated. Treat the absolute numbers
accordingly; the comparisons above are all within one checkpoint.

## Limitations

- 10–20 hand-written examples per experiment, labelled by the author. Enough to show a
  capability exists or a class is dead; not enough to measure accuracy.
- Every threshold is eyeballed, never fitted on a validation set. Guard 1's clear/ambiguous
  gap spans 0.49–0.75, so any cutoff inside it gives identical results.
- The guard dataset confounds two properties: every clear example is both well-specified
  *and* determinate. Separating sufficiency from alignment needs cases where specificity
  and correctness vary independently — which is why finding 2 indicts the experiment as
  much as the model.
- One guard example (`"Translate this paragraph into French."`, no paragraph present) is
  arguably mislabelled as well-specified, and guard 3 is the run it penalises.
