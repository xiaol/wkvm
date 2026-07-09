# Both: Quality + Concurrency Target for Gemma Recurrent Mode

The plain ring result is not enough. It wins concurrency, but it loses evicted
facts. The defensible target is `routed-span-m64`: bounded recurrent memory with
span-atomic content routing.

## What Is Already Measured

Same Gemma-4-E4B-it model, RTX 4090.

Quality, from `quality_grid.md` and `quality_nll.md`:

| mode | recall overall | t1 needle | t2 multi-key | t3 aggregate | deep NLL delta vs full |
|---|---:|---:|---:|---:|---:|
| full KV | 0.95 | 1.00 | 1.00 | 0.86 | 0.00 |
| ring | 0.06 | 0.07 | 0.07 | 0.04 | +0.3 to +1.4 nats |
| banked | 0.45 | 0.80 | 0.33 | 0.21 | still large |
| routed-value-m64 | 0.79 | 1.00 | 0.51 | 0.86 | still large |
| routed-span-m64 | 0.90 | 1.00 | 0.89 | 0.81 | +0.00 to +0.35 nats |

Concurrency/capacity, measured or derived:

| mode | per-session memory | 4k capacity | 16k capacity | 32k capacity |
|---|---:|---:|---:|---:|
| ring | 36.3 MiB | 96 measured green | 96 measured green | flat by design |
| routed-span-m64 | ~88 MiB | ~43 derived green | ~43 derived green | ~43 derived green |
| vLLM full KV | 84 MiB @4k, 276 MiB @16k, ~552 MiB @32k | 38 measured | 9 measured | ~4 derived |
| SGLang full KV | pool-limited on this stack | 6 measured | 1 measured | not fitting |

The routed-span capacity estimate uses the measured ring ladder's non-cache
baseline and the documented routed-span slot size. It is not yet a native
scheduler/server measurement.

## Justified Claim

`routed-span-m64` is the current "both" candidate:

- It preserves the flat-memory shape.
- It recovers most synthetic long-recall quality: 0.90 overall vs full-KV 0.95.
- It keeps enough concurrency to matter: about 43 long sessions under the old
  green memory line, versus vLLM's 9 at 16k and about 4 at 32k.
- It still has an honest gap: it is not exact full attention, and the remaining
  t2 failures are sibling eviction under fixed slot budget.

## Acceptance Criteria For A Defensible Native Result

1. No patched HF serving path:
   `rg gemma_recurrent_poc wkvm/` must return no results.
2. Native runner:
   `wkvm.engine` can run Gemma `routed-span-m64` through the scheduler with
   prompt prefill and batched decode.
3. Quality gates:
   - pre-eviction NLL delta <= 1e-3,
   - recall overall >= 0.90 on the existing grid,
   - t2 multi-key >= 0.89,
   - routed-span NLL curve no worse than the current table.
4. Concurrency gates:
   - at least 40 resident 16k sessions under the green memory budget,
   - at least 40 resident 32k sessions under the green memory budget,
   - aggregate decode reported at B=8,16,32,40 with no replicated-session caveat.
5. Baselines in the same report:
   - vLLM full-KV capacity/throughput at 16k and 32k,
   - SGLang full-KV where it fits,
   - token-level examples showing where recurrent output matches or differs.

## Next Engineering Step

Implement the native Gemma recurrent runner with `routed-span-m64` first, not
plain ring. Plain ring is a useful ablation, but it cannot support the "both"
claim.
