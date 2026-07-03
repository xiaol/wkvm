# M2 engine bench — steady-state decode from the engine loop

*RTX 4090 24GB, 2026-07-03, torch 2.11 cu130 + fla, bf16 weights / fp32 wkv state.
Every timed token flows through `Engine.step()`: `scheduler.schedule()` →
gather/forward/scatter → batched sample → `update_from_output()`. All B slots
decoding, nobody finishes mid-measurement (prompt 32, 64 timed steps, 8 warmup).
Reproduce: `python experiments/m2_engine_bench.py [--model PATH] --graph`.*

## RWKV-7 World 1.5B (24 layers, d=2048) — 12.19 MiB state/slot

| B | tok/s (eager) | tok/s (graphed) | state MiB total | peak VRAM GiB |
|---|---|---|---|---|
| 1 | 67 | 195 | 24 | 2.96 |
| 8 | 516 | 1,084 | 110 | 3.19 |
| 32 | 1,890 | 3,565 | 402 | 4.36 |
| 64 | 3,592 | 5,274 | 792 | 5.92 |
| 128 | 6,345 | 6,886 | 1,572 | 9.84 |
| 256 | 7,913 | 8,077 | 3,132 | 19.02 |

## RWKV-7 World 191M (12 layers, d=768) — 2.29 MiB state/slot

| B | tok/s (eager) | tok/s (graphed) | state MiB total | peak VRAM GiB |
|---|---|---|---|---|
| 1 | 132 | 680 | 5 | 0.45 |
| 8 | 1,038 | 4,160 | 21 | 0.46 |
| 32 | 3,991 | 15,299 | 75 | 0.69 |
| 64 | 7,738 | 23,471 | 149 | 0.87 |
| 128 | 14,506 | 32,338 | 295 | 1.77 |
| 256 | 26,707 | 38,856 | 587 | 2.94 |

## Notes

- **Graphed** = the decode-step *model forward* captured in one CUDA graph per
  batch-size bucket (`GraphedRunner` in `m2_engine_bench.py`), replayed while
  batch composition is unchanged; per-layer state gather/scatter and the
  scheduler remain eager python. Verified token-identical to the eager path
  (48 greedy tokens × 3 real prompts through the engine, both runners).
- Graph capture worked first try: fla's fused-recurrent triton kernels and the
  seeded-`Cache` forward are capture-safe under `torch.cuda.graph` with static
  input/state tensors; output-state addresses are stable in the graph pool.
- Speedup is python-launch-overhead-bound, so it decays with B: 2.9× at B=1,
  1.05× at B=256 (1.5B). At B=1 the graphed step is ~5.1 ms of which ~2.6 ms
  is the replay — the rest is the eager gather/scatter (72 `index_select` +
  72 `index_copy_` at 24 layers) plus scheduler python. Folding gather/scatter
  into the capture (or batching it across layers) is the obvious next win.
- Peak VRAM is for the `--graph` run and includes the graph's static input
  tensors (a second `[B, ...]` state copy, ~3.1 GiB at 1.5B/B=256) and the
  graph memory pool; eager-only peaks are correspondingly lower.
- vs Albatross (RWKV-7 **2.9B** fp16, hand-tuned sm_89 CUDA, static batch, no
  scheduler — see docs/COMPARISON.md §3): 195 / 840 / 3,595 / 6,042 / 10,158 /
  12,707 tok/s at B = 1/8/32/64/128/256. Our graphed 1.5B ladder sits at
  ~0.6–1.0× Albatross's *2.9B* numbers per batch point despite running a model
  half the size — i.e. per-parameter we are roughly 2× off their kernel
  efficiency, which is the expected price of python-loop + generic fla kernels
  vs hand-tuned CUDA, before any fused gather/scatter or whole-step capture.
