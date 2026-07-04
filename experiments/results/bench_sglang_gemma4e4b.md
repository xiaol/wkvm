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

---

# UPDATE 2026-07-05: FIXED — SGLang now serves this model (attempt 9)

The DNF above was resolved by three root-cause fixes + one environment shim:

1. **Corrupted `nvidia-cutlass-dsl` install** (the MLIR ICE): mixed-version package files — the NVIDIA/cutlass#3132 failure mode, here almost certainly caused by two parallel `uv pip install` processes contending on the shared uv cache lock during initial setup. Proven by reproducing the byte-identical ICE **on CPU** (`wkvm_bench/repro_rmsnorm_ice.py`), then clean-reinstalling the *same* version (4.5.2) — after which all 10 gemma4 norm-kernel variants compile (`wkvm_bench/verify_norm_kernels_cpu.py`).
2. **Why the kill-switch never worked**: `sgl_kernel/elementwise.py` imports `flashinfer.norm` behind its own module-level `_has_flashinfer` flag — `SGLANG_IS_FLASHINFER_AVAILABLE=false` gates sglang's paths but not sgl_kernel's.
3. **tvm-ffi JIT** (attempt 8's new failure): sgl_kernel's fused_rope JIT hit the GCC15/glibc `rsqrtf` exception-spec clash — fixed with the existing CUDA-glibc shim (`CPATH=.../tools/cuda-glibc-shim`).
4. Config retained from the DNF forensics: mem-fraction 0.88, `max_running_requests=64`, `SGLANG_SWA_EVICTION_INTERVAL=128`, decode graphs `full`, prefill graphs `disabled` (Gemma4TextModel `.model` attribute bug).

## Measured (attempt 9, RTX 4090, triton attention backend)

- KV pool: **25,360 tokens** → 6 concurrent 4k sessions, 1 at 16k. (vLLM: 161,584 tokens / 38 / 9 — SGLang's run carries multimodal tower weights that vLLM's `limit_mm_per_prompt={image:0,audio:0}` freed, plus the SWA/full dual-pool split.)
- Aggregate decode: **410 tok/s** at N=64/4k (queue-limited over 6 truly-concurrent), **68 tok/s** at N=8/16k. The N=8/4k decode figure in the raw table is a timing artifact (t₁ ≈ t₁₂₈ under queueing) — disregard that cell.
- Raw table: `wkvm_bench/results_sglang.md`; logs `wkvm_bench/tmp/bench_sglang_run8.log`, `run9.log`.

Caveats as in COMPARISON.md: triton attention (not SGLang's best backend), no prefill graphs, mm towers resident. The engineering lesson stands strengthened: four independent JIT/startup-path dependencies each cost one debugging round on a fresh-but-real deployment stack.
