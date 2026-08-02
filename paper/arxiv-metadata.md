# Draft arXiv metadata

This file is a staging aid, not authorization to submit. Confirm the author
identity, affiliation, subject classes, and paper license in the arXiv UI.

## Title

WKVM: Treating Model State as a First-Class Serving Object — State-Native
Inference for Recurrent and Hybrid Language Models

## Authors

xiaol **[confirm the publication name and add affiliation/ORCID if desired]**

## Abstract

Contemporary language-model servers are organized around a key–value (KV)
cache whose size grows with retained token history. Recurrent and linear
attention models expose different serving physics: each request carries a
fixed-size execution state, independent of context length. We present WKVM, a
prototype that makes this state the primary allocation and lifecycle object.
The native RWKV-7 path combines typed state families, all-or-nothing slot
admission, continuous batching over computed-token gaps, dense GPU state banks,
and named versioned snapshots that can be restored, forked, persisted, and
mutated with lineage. A separate Gemma guest path retains a bounded set of
actual KV spans selected by value-vector routing; it is approximate and is not
an RWKV recurrence. On one RTX 4090, a single-run RWKV-7 1.5B experiment reaches
8,077 tokens/s at batch 256 with 12.19 MiB state per request. A 191M
state-store demonstration creates 2,000 snapshots, measures 8.2 ms median warm
restore, and reproduces 16/16 interrupted continuations. For bounded Gemma,
synthetic recall averages 0.90 versus 0.95 for full KV. In three clean A800
runs at batch 64 and 16K input tokens, WKVM delivers 4.47x the comparable vLLM
decode rate but only 0.79x its end-to-end throughput because prefill dominates.
The results support fixed resident state and explicit state lifecycle
management; they do not support quality equivalence or a universal speedup.

## Suggested classification

- Primary: `cs.DC` (Distributed, Parallel, and Cluster Computing)
- Cross-list: `cs.LG` (Machine Learning)
- Alternative primary if the final emphasis changes: `cs.PF` (Performance)

## Comments

Technical report. 10 pages, 1 figure, 5 tables. Code and artifact summaries:
https://github.com/xiaol/wkvm

## License

**Choose explicitly in arXiv.** The paper license is separate from the code
license. The repository currently declares Apache-2.0 in `pyproject.toml` but
also says the top-level `LICENSE` file is pending; resolve that inconsistency
before advertising the code as Apache-licensed.

## Submission blockers

- Confirm the author’s publication name and ownership/approval of the text.
- Choose the arXiv distribution license.
- Confirm the primary category and any endorsement requirement.
- Review every promoted claim against `paper/CLAIMS.md`.
- Inspect the final PDF and source archive produced by `make arxiv`.
