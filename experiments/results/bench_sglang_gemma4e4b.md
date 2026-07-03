# SGLang bench: gemma-4-E4B-it — DNF on this stack (2026-07-03)

**Result: could not serve this model on this machine after 7 documented attempts.** Not benchmarked; recorded as an operational finding, not an architectural verdict — SGLang would likely run on a cu126/torch-2.9 stack.

Install: sglang 0.5.14 (required `--prerelease=allow` pin: plain `uv pip install "sglang[all]"` silently backtracks to 0.5.9, which lacks gemma4), torch 2.11.0+cu130, flashinfer 0.6.12, python 3.12, RTX 4090, driver 595.71.

## Attempt chain

| # | Config | Failure |
|---|---|---|
| 1 | defaults, mem-fraction 0.82 | `pool_configurator`: SWA pool cap (86,145 tok, 5.75 GiB) leaves no room for full-KV pool (1.58 GiB available) |
| 2 | mem-fraction 0.88 | same, 3.21 GiB available — SWA cap unchanged (scales with max_running_requests) |
| 3 | + max_running_requests 64, SGLANG_SWA_EVICTION_INTERVAL=128 | pools OK → crash in `tc_piecewise` prefill CUDA-graph compile: `'Gemma4TextModel' object has no attribute 'model'` (sglang bug, gemma4 text path) |
| 4 | + decode graphs `full`, prefill graphs `disabled` | flashinfer cute-DSL rmsnorm JIT: `nvidia-cutlass-dsl` internal compiler error — MLIR `'llvm.mlir.global_dtors' op requires attribute 'data'` |
| 5 | + `--attention-backend triton` | same ICE (norm path, not attention) |
| 6 | + `SGLANG_IS_FLASHINFER_AVAILABLE=false` (all-triton) | back to pool error under triton memory layout |
| 7 | + max_running_requests 32, interval 64, mem-fraction 0.90 | pools OK → cutlass ICE again: a cutlass-DSL JIT path survives the flashinfer gate |

## Diagnosis

Two independent blockers stacked:
1. **Hybrid-model pool configurator friction**: on a 24GB card, the SWA pool sizing (driven by `max_running_requests` × window math for the 35 sliding layers) repeatedly starved the full-KV pool for the 4 global layers; needed manual joint tuning of three knobs just to allocate.
2. **`nvidia-cutlass-dsl` is broken on this toolchain** (CUDA 13.1 / MLIR mismatch): flashinfer's cute-DSL kernels (rmsnorm and at least one more path not gated by `SGLANG_IS_FLASHINFER_AVAILABLE`) hit an internal compiler error. vLLM 0.24.0 on the *same venv-toolchain family* (torch 2.11+cu130, flashinfer 0.6.12) served this model first try at mem-fraction 0.82 — it evidently does not route through the cute-DSL JIT for this model.

## Comparison-relevant read

For the engine-comparison doc: SGLang's architecture (radix cache, overlap scheduler) is not implicated here; what failed is the breadth surface — model-specific graph-backend code paths (`Gemma4TextModel` attribute bug) and a JIT-compiled kernel dependency chain deep enough that no user-facing flag fully disables it. A from-scratch engine's lesson (docs/ANGLE.md §3 "refuse" list): every JIT dependency on the startup path is a deployment risk multiplier.
