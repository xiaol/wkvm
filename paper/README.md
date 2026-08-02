# WKVM technical report

This directory contains an arXiv-oriented technical report and an explicit
claim audit. The report deliberately separates the exact native RWKV-7 path
from the approximate Gemma retained-KV routed-span path.

Rendered report: [wkvm-technical-report.pdf](wkvm-technical-report.pdf)

## Build

```bash
cd paper
make pdf
```

The PDF is written to `paper/build/main.pdf`.

Refresh the committed PDF after editing the report:

```bash
make publish-pdf
```

Create the source archive used for an arXiv upload:

```bash
make arxiv
```

This writes `paper/build/wkvm-arxiv-source.tar.gz`. The archive contains only
`main.tex` and `references.bib`; arXiv can run BibTeX from those sources.

## Files

- `main.tex` — report source.
- `references.bib` — bibliography.
- `CLAIMS.md` — evidence, caveats, and excluded claims.
- `arxiv-metadata.md` — draft submission metadata and human decisions still
  required.
- `Makefile` — local PDF and source-archive targets.

## Before submission

1. Confirm or replace the author handle `xiaol` with the intended publication
   name, affiliation, and ORCID.
2. Resolve the repository’s missing top-level `LICENSE` file before making a
   code-license statement.
3. Choose the paper’s arXiv license; code and paper licenses are independent.
4. Review `CLAIMS.md`, especially the single-run and cross-semantics caveats.
5. Open `build/main.pdf`, check tables/references, and update page/table/figure
   counts in `arxiv-metadata.md`.
6. Upload `build/wkvm-arxiv-source.tar.gz` through the author’s arXiv account.

Actual submission requires the author’s identity, license choice, account, and
possibly category endorsement. Those are intentionally not inferred from the
Git repository.
