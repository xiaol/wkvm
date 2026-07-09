# Native Gemma Throughput Frontier

Gemma-4-E4B-it, one RTX 4090, 13,824-token context per session, 128 decode tokens per session, distinct staggered prompts. Green means the run stayed under a 19 GiB memory line with 1 GiB headroom.

These are native wkvm scheduler/cache runs, not native-kernel transformer runs. The server loads `Gemma4ForCausalLM` from Transformers, and the runner calls that model for prefill/decode with `NativeGemmaRoutedCache`.

| run | B | green | aggregate decode tok/s | p50 latency | p95 latency | peak reserved | max model rows | max model bytes | decode calls | fallback calls | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| scheduler-only native baseline | 32 | yes | 30.822 | 134.201s | 135.138s | 17.639 GiB | 32 scheduled | n/a | n/a | n/a | [`json`](native_gemma_routed_span_concurrency.json) |
| padded decode, row cap 4 | 32 | yes | 50.482 | 83.334s | 84.540s | 17.623 GiB | 4 | n/a | 1016 | 0 | [`json`](native_gemma_batched_decode_ctx13824_out128_b32_micro4.json) |
| padded decode, byte cap 350M | 32 | yes | 57.869 | 73.067s | 74.307s | 17.963 GiB | 6 | 349,765,632 | 762 | 0 | [`json`](native_gemma_bytecap_ctx13824_out128_b32_350m.json) |
| padded decode, byte cap 370M | 32 | yes | 57.933 | 72.935s | 74.122s | 17.963 GiB | 6 | 349,765,632 | 762 | 0 | [`json`](native_gemma_bytecap_ctx13824_out128_b32_370m.json) |
| padded decode, byte cap 370M, instrumented | 32 | yes | 54.554 | 77.305s | 78.452s | 17.963 GiB | 6 | 349,765,632 | 762 | 0 | [`json`](native_gemma_bytecap_ctx13824_out128_b32_370m_instrumented.json) |
| length-bucketed padded decode, byte cap 370M | 32 | yes | 54.569 | 77.460s | 78.718s | 17.865 GiB | 6 | 348,094,464 | 762 | 0 | [`json`](native_gemma_length_bucketed_ctx13824_out128_b32_370m.json) |
| padded decode, byte cap 370M, retained workspace | 32 | no | 54.408 | 77.604s | 78.852s | 20.195 GiB | 6 | 349,765,632 | 762 | 0 | [`json`](native_gemma_workspace_ctx13824_out128_b32_370m.json) |
| padded decode, byte cap 400M | 32 | no | 57.918 | 72.974s | 74.152s | 18.133 GiB | 7 | 399,917,056 | 688 | 0 | [`json`](native_gemma_bytecap_ctx13824_out128_b32_400m.json) |
| padded decode, byte cap 470M | 32 | no | 59.763 | 70.908s | 72.139s | 19.123 GiB | 8 | 466,354,176 | 508 | 0 | [`json`](native_gemma_bytecap_ctx13824_out128_b32_470m.json) |
| padded decode, row cap 16 | 32 | no | 66.174 | 64.231s | 65.419s | 20.600 GiB | 16 | 932,708,352 | 254 | 0 | [`json`](native_gemma_reserved_decode_ctx13824_out128_b32_micro16.json) |

## Hugging Face Transformers Baseline

Same model, prompt builder, GPU, 13,824-token context/session, and 128 decode tokens/session. This baseline uses plain `Gemma4ForCausalLM` with HF full-KV cache semantics. Prefill is chunked at 2,048 tokens to avoid turning the baseline into a one-shot full-context allocation test.

| run | B | green | aggregate decode tok/s | p50 latency | p95 latency | peak reserved | source |
|---|---:|---:|---:|---:|---:|---:|---|
| HF Transformers serial, chunked prefill | 1 | yes | 26.643 | 6.299s | 6.299s | 15.486 GiB | [`json`](hf_gemma_serial_ctx13824_out128_b1_chunk2048.json) |
| HF Transformers batched, chunked prefill | 1 | yes | 26.537 | 6.288s | 6.288s | 15.518 GiB | [`json`](hf_gemma_batched_chunked_ctx13824_out128_ladder.json) |
| HF Transformers batched, chunked prefill | 2 | yes | 52.552 | 7.457s | 7.457s | 16.879 GiB | [`json`](hf_gemma_batched_chunked_ctx13824_out128_ladder.json) |
| HF Transformers batched, chunked prefill | 4 | no | 86.174 | 12.088s | 12.088s | 20.312 GiB | [`json`](hf_gemma_batched_chunked_ctx13824_out128_ladder.json) |

Readout:

- Best green B=32 run is now **57.933 aggregate decode tok/s**, up from the older native baseline's **30.822 tok/s**.
- The best green HF Transformers baseline measured here is **52.552 aggregate decode tok/s at B=2**. wkvm-native is only **10.2% faster in green aggregate decode throughput**, but keeps **32** long sessions resident under the same green line instead of **2**.
- HF batched B=4 reaches **86.174 tok/s** but is over the green memory line at **20.312 GiB**. The closest wkvm over-memory row is row cap 16 at **66.174 tok/s** and **20.600 GiB**, so current native wkvm should not be claimed faster than HF when both are allowed to exceed the 19 GiB line.
- The red frontier shows the current memory/throughput tradeoff: row cap 16 reaches **66.174 tok/s** but reserves **20.600 GiB**.
- Length bucketing was tested and rejected for now: it reduced peak reserved memory slightly but regressed aggregate decode throughput from **57.933** to **54.569 tok/s**.
- A retained padded-decode workspace was tested and rejected as a default: it reused **18,172** layer buffers after **116** allocations, but peak reserved memory rose to **20.195 GiB** and throughput stayed at **54.408 tok/s**. This confirms allocator reuse alone does not close the production gap under the green memory line.
- All padded decode frontier runs completed **32/32** wkvm requests with **0 fallback decode calls** and **4,064 padded decode rows**.
- The instrumented 370M row materialized **231,579,315,352 bytes** of temporary HF-compatible padded KV/mask tensors across the run, with a largest decode model call of **349,881,600 bytes**. That is the immediate cost to eliminate with a native backend that consumes routed spans directly.
- This still is not vLLM/SGLang parity: wkvm owns admission, slots, routed-span cache shape, and decode microbatch planning, while HF Gemma still owns the transformer forward kernels.

Source-code comparison:

- wkvm loads HF Gemma in [`wkvm/gemma_server.py`](../../wkvm/gemma_server.py), then calls the HF model in [`wkvm/runner/gemma_runner.py`](../../wkvm/runner/gemma_runner.py) for prefill and decode.
- wkvm batches decode by merging per-request routed caches into exact or padded temporary cache objects, then commits padded decode updates back into per-request caches.
- vLLM schedules with token budgets, prefix-cache awareness, and preemption in `vllm/v1/core/sched/scheduler.py`; allocates paged KV blocks in `vllm/v1/core/kv_cache_manager.py`; maps logical tokens to physical KV slots in `vllm/v1/worker/gpu_model_runner.py`; and dispatches through attention backends and CUDA graph paths.
- SGLang keeps a persistent running batch in `sglang/srt/managers/scheduler.py`, allocates decode KV through `ScheduleBatch.prepare_for_decode()` and `alloc_for_decode()`, stores request-to-token mappings in `ReqToTokenPool`, and routes attention through backend calls from `RadixAttention`.

Next production gap:

The immediate bottleneck is the padded HF-compatible cache materialization path. Retaining reusable padded buffers is available as an experimental opt-in, but it made the B=32 frontier red on this GPU. To become comparable to vLLM/SGLang as an engine, wkvm needs a true backend path that consumes routed-span state directly, plus graph-captured decode shapes. That is separate from the already-measured native scheduler/cache improvement above.
