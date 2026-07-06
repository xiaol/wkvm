#!/usr/bin/env python
"""PoC-1: "recurrent mode" serving for gemma-4-E4B-it on one RTX 4090.

Demonstrates that capping the growing-KV layers (the full-attention layers that
own KV: derived from config, e.g. layers 5/11/17/23 of gemma-4-E4B) with a fixed
sink+ring cache gives FLAT memory and FLAT decode speed vs context length, while
stock full KV grows linearly. A needle-recall probe documents the quality gap the
ring introduces (to be closed by the state bank in PoC-2).

Design notes (verified against transformers 5.9.0 in the HRM-Text venv):
- v5 caches are per-layer objects. DynamicCache(config=...) builds one layer per
  NON-shared decoder layer (num_kv_shared_layers tail layers own no cache) using
  LAYER_TYPE_CACHE_MAPPING; sliding layers already get a bounded
  DynamicSlidingWindowLayer. We replace only the full_attention owned layers with
  SinkRingLayer below.
- KV sharing: the last owned layer of each type (store_full_length_kv) publishes
  its post-cache-update KV into shared_kv_states[layer_type]; the shared tail
  layers consume that. So bounding layer 23's cache automatically bounds the
  shared full-attention layers 29/35/41 too.
- Masks: create_causal_mask() builds ONE mask for the whole full_attention group
  from past_key_values.is_sliding.index(False) -> our layer 5. It asks the layer
  for (kv_length, kv_offset) via get_mask_sizes(). We report kv_offset such that
  imputed KV positions run contiguously up to the current position: for a causal
  (full) mask every cached slot is then visible to every query (correct - all
  cached tokens ARE in the past), and the last Q slots line up exactly with the
  query positions, so intra-chunk causality during chunked prefill is exact.
- StreamingLLM caveat: keys are stored post-RoPE, so evicting middle positions
  keeps absolute positions valid on the kept sink/ring entries; no re-rotation is
  needed. The model simply no longer sees the evicted middle - that is the lossy
  part that the PoC-2 state bank will compensate.

Run (offline, no HF_HOME override):
  HF_HUB_OFFLINE=1 /home/xiaol/X/HRM-Text/.venv/bin/python \
      experiments/gemma_recurrent_poc.py bench
"""

import argparse
import gc
import math
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# ~1.2 GiB less fragmentation at B>=96 for ~6% decode throughput — the right
# trade under a shared-GPU budget (override via env to compare).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from transformers import AutoConfig, AutoTokenizer
from transformers.cache_utils import DynamicCache, DynamicLayer, DynamicSlidingWindowLayer

MODEL_CANDIDATES = [
    "/run/media/xiaol/B214449214445C0B/models/gemma/gemma-4-E4B-it",
    "google/gemma-4-E4B-it",
]
FALLBACK_E2B = "google/gemma-4-E2B-it"

FILLER = (
    "The old lighthouse keeper walked along the shore every morning, checking the "
    "tide tables and noting the weather in his worn leather journal. Gulls wheeled "
    "overhead while fishing boats returned with the dawn catch, their hulls heavy "
    "and their crews tired but satisfied after a long night on the water. "
)
NEEDLE = "The secret code is BLUE-742. Remember it carefully. "
QUESTION = (
    "What is the secret code mentioned earlier in this text? "
    "Answer with just the code."
)


# --------------------------------------------------------------------------- #
# Sink+ring cache layer
# --------------------------------------------------------------------------- #
class SinkRingLayer(DynamicLayer):
    """Bounded KV layer: keeps the first `sink` tokens ever seen plus a ring of
    the most recent `window` tokens. Everything in between is evicted.

    is_sliding stays False so masking_utils keeps this layer in the
    full_attention mask group (create_causal_mask picks the first non-sliding
    layer's get_mask_sizes to shape the group mask).

    get_mask_sizes() imputes contiguous positions [cum - stored, cum + Q) onto
    the stored slots. Sink slots therefore masquerade as recent positions, which
    is harmless under a *causal* mask (they are unconditionally in the past) and
    keeps the returned KV length consistent with the mask width. Keys are cached
    post-RoPE (StreamingLLM-style), so kept entries stay positionally valid.
    """

    is_sliding = False

    def __init__(self, sink: int = 16, window: int = 1024):
        super().__init__()
        self.sink = int(sink)
        self.window = int(window)
        self.cumulative_length = 0

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        self.cumulative_length += key_states.shape[-2]

        full_keys = torch.cat([self.keys, key_states], dim=-2)
        full_values = torch.cat([self.values, value_states], dim=-2)

        cap = self.sink + self.window
        if full_keys.shape[-2] > cap:
            # First `sink` positions ever seen + last `window` positions.
            self.keys = torch.cat(
                [full_keys[..., : self.sink, :], full_keys[..., -self.window :, :]], dim=-2
            )
            self.values = torch.cat(
                [full_values[..., : self.sink, :], full_values[..., -self.window :, :]], dim=-2
            )
        else:
            self.keys, self.values = full_keys, full_values

        # Return the *un-evicted* states for this step's attention (the current
        # chunk always attends to everything still cached + itself).
        return full_keys, full_values

    def get_mask_sizes(self, query_length: int):
        stored = 0 if self.keys is None or not self.is_initialized else self.keys.shape[-2]
        kv_length = stored + query_length
        kv_offset = self.cumulative_length - stored
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.sink + self.window


class BankedRingLayer(SinkRingLayer):
    """PoC-2: sink+ring plus a capacity-bounded multi-state bank of evicted KV.

    Evicted ring tokens accumulate in a pending buffer; every `seg` evicted
    tokens are folded into one bank *segment state* consisting of `reps`
    pseudo-KV slots: slot 0 is the segment summary (mean key, mean value) and
    slots 1..reps-1 are the segment's top *representatives* — the exact (K,V)
    of the tokens whose keys are most novel w.r.t. the segment mean (cosine,
    averaged over KV heads). Keys are post-RoPE, but full-attention layers use
    partial_rotary_factor=0.25, so 3/4 of key dims are position-free and the
    novelty score is dominated by content, not position.

    Readout is option (a) of the PoC-2 spec: bank slots are ordinary KV entries
    inside self.keys/values ([sink | bank | pending | ring], chronological), so
    they enter stock softmax attention through the exact same imputed-position
    causal mask as the ring — no modeling-code changes.

    Capacity: when the bank exceeds k_states segments, the two most similar
    adjacent segments merge (count-weighted mean summary; representatives
    re-selected by novelty against the merged mean) — DLA-style capacity-
    bounded adjacent merging, so footprint stays flat at any context length.
    Batch ops (repeat/reorder) are not supported: PoC quality path is B=1.
    """

    def __init__(self, sink=16, window=1024, k_states=16, seg=512, reps=8,
                 coord=None, is_leader=True):
        super().__init__(sink, window)
        self.k_states, self.seg, self.reps = int(k_states), int(seg), int(reps)
        self._segs: list[dict] = []
        self._folded = 0  # evicted tokens folded into segments so far
        # Cross-layer selection sharing: deep-layer keys/values are not needle-
        # selective (verified empirically: novelty picks the needle tokens
        # perfectly at layers 5/11 and fails at 17/23), so the shallowest banked
        # layer is the *leader* whose representative indices and merge decisions
        # are logged to `coord` and replayed by follower layers. Layers run in
        # order and share the identical eviction schedule, so the op log aligns.
        self.coord, self.is_leader = coord, is_leader
        self._op_cursor = 0

    def _decide(self, kind, compute):
        """Leader computes a structural decision and logs it; followers replay."""
        if self.coord is None or self.is_leader:
            op = (kind, *compute())
            if self.coord is not None:
                self.coord.append(op)
        else:
            op = self.coord[self._op_cursor]
            assert op[0] == kind, f"bank op log desync: {op[0]} != {kind}"
        self._op_cursor += 1
        return op

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        B, H, _, D = key_states.shape
        empty = lambda: torch.zeros(B, H, 0, D, dtype=self.dtype, device=self.device)
        self._sink_k, self._sink_v = empty(), empty()
        self._ring_k, self._ring_v = empty(), empty()
        self._pend_k, self._pend_v = empty(), empty()

    def _fold_segment(self):
        S = self.seg
        cut_k, cut_v = self._pend_k[:, :, :S], self._pend_v[:, :, :S]
        self._pend_k, self._pend_v = self._pend_k[:, :, S:], self._pend_v[:, :, S:]
        mean_k, mean_v = cut_k.mean(2, keepdim=True), cut_v.mean(2, keepdim=True)
        rep_k = rep_v = None
        if self.reps > 1:
            def compute():
                cos = torch.nn.functional.cosine_similarity(
                    cut_k.float(), mean_k.float(), dim=-1)      # [B, H, S]
                novelty = (1.0 - cos).mean(1)                    # [B, S]
                return (novelty[0].topk(min(self.reps - 1, S)).indices.sort().values,)
            _, idx = self._decide("fold", compute)
            rep_k, rep_v = cut_k[:, :, idx], cut_v[:, :, idx]
        start_abs = self.sink + self._folded
        rep_abs = (idx + start_abs).tolist() if self.reps > 1 else []
        self._folded += S
        self._segs.append(dict(mk=mean_k, mv=mean_v, rk=rep_k, rv=rep_v, n=S,
                               span=(start_abs, start_abs + S), rep_abs=rep_abs))

    def _merge_once(self):
        """Merge the two most similar adjacent segments (novelty re-selection).
        Pair choice and kept-representative indices come from the leader."""
        def compute():
            sims = [
                torch.nn.functional.cosine_similarity(
                    self._segs[j]["mk"].float(), self._segs[j + 1]["mk"].float(), dim=-1
                ).mean().item()
                for j in range(len(self._segs) - 1)
            ]
            j = max(range(len(sims)), key=sims.__getitem__)
            keep = None
            if self.reps > 1:
                a, b = self._segs[j], self._segs[j + 1]
                n = a["n"] + b["n"]
                mk = (a["mk"] * a["n"] + b["mk"] * b["n"]) / n
                pool_k = torch.cat([a["rk"], b["rk"]], dim=2)
                cos = torch.nn.functional.cosine_similarity(pool_k.float(), mk.float(), dim=-1)
                novelty = (1.0 - cos).mean(1)
                keep = novelty[0].topk(min(self.reps - 1, pool_k.shape[2])).indices.sort().values
            return j, keep

        _, i, keep = self._decide("merge", compute)
        a, b = self._segs[i], self._segs[i + 1]
        n = a["n"] + b["n"]
        mk = (a["mk"] * a["n"] + b["mk"] * b["n"]) / n
        mv = (a["mv"] * a["n"] + b["mv"] * b["n"]) / n
        rk = rv = None
        rep_abs = []
        if self.reps > 1:
            pool_k = torch.cat([a["rk"], b["rk"]], dim=2)
            pool_v = torch.cat([a["rv"], b["rv"]], dim=2)
            pool_abs = a["rep_abs"] + b["rep_abs"]
            rk, rv = pool_k[:, :, keep], pool_v[:, :, keep]
            rep_abs = [pool_abs[j] for j in keep.tolist()]
        self._segs[i : i + 2] = [dict(mk=mk, mv=mv, rk=rk, rv=rv, n=n,
                                      span=(a["span"][0], b["span"][1]), rep_abs=rep_abs)]

    def _materialize(self):
        parts_k, parts_v = [self._sink_k], [self._sink_v]
        for s in self._segs:
            parts_k.append(s["mk"]); parts_v.append(s["mv"])
            if s["rk"] is not None:
                parts_k.append(s["rk"]); parts_v.append(s["rv"])
        parts_k += [self._pend_k, self._ring_k]
        parts_v += [self._pend_v, self._ring_v]
        self.keys = torch.cat([p for p in parts_k if p.numel()], dim=2)
        self.values = torch.cat([p for p in parts_v if p.numel()], dim=2)

    def n_bank_slots(self) -> int:
        return sum(1 + (0 if s["rk"] is None else s["rk"].shape[2]) for s in self._segs)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert key_states.shape[0] == 1, "BankedRingLayer is B=1 (PoC quality path)"
        # This step's attention sees everything stored so far + the new tokens.
        ret_k = torch.cat([self.keys, key_states], dim=-2) if self.keys.numel() else key_states
        ret_v = torch.cat([self.values, value_states], dim=-2) if self.values.numel() else value_states
        self.cumulative_length += key_states.shape[-2]

        # Bookkeeping: new tokens -> ring; fill sink; overflow -> pending -> bank.
        rk = torch.cat([self._ring_k, key_states], dim=2)
        rv = torch.cat([self._ring_v, value_states], dim=2)
        deficit = self.sink - self._sink_k.shape[2]
        if deficit > 0:
            take = min(deficit, rk.shape[2])
            self._sink_k = torch.cat([self._sink_k, rk[:, :, :take]], dim=2)
            self._sink_v = torch.cat([self._sink_v, rv[:, :, :take]], dim=2)
            rk, rv = rk[:, :, take:], rv[:, :, take:]
        if rk.shape[2] > self.window:
            cut = rk.shape[2] - self.window
            self._pend_k = torch.cat([self._pend_k, rk[:, :, :cut]], dim=2)
            self._pend_v = torch.cat([self._pend_v, rv[:, :, :cut]], dim=2)
            rk, rv = rk[:, :, cut:], rv[:, :, cut:]
        self._ring_k, self._ring_v = rk, rv
        while self._pend_k.shape[2] >= self.seg:
            self._fold_segment()
        while len(self._segs) > self.k_states:
            self._merge_once()
        self._materialize()
        return ret_k, ret_v

    def batch_repeat_interleave(self, repeats: int):
        raise NotImplementedError("BankedRingLayer does not support batch replication")


class RoutedBankLayer(SinkRingLayer):
    """Training-free ROUTED bank (SelectingMemory/Raven idea, zero learned params).

    Instead of temporal segments (BankedRingLayer), evicted tokens are routed to
    M persistent slots by online spherical clustering of their features at the
    LEADER layer (deep-layer keys smear — PoC-2 finding), with assignments and
    rep-retention decisions replayed on follower layers via the same op-log.

    Per slot: running mean-K/mean-V summary (fp32 accumulators) + up to `reps`
    exact post-RoPE (K,V) representatives. A slot mixes tokens from arbitrary
    positions — safe for the pseudo-KV readout, because exact reps carry their
    own RoPE'd positions and the causal mask makes every cached slot visible;
    nothing re-derives position from slot order.

    Routing features (`route_on`): "key" = raw keys; "resid" = keys minus the
    running global key mean (strips the shared syntactic-template component —
    the known t2 risk); "value" = value vectors (no RoPE at all).
    Assignment: cosine argmax to M centroids; touched centroids get an EMA
    update (momentum 0.9); untouched slots never decay (the Raven property).
    Init: deterministic farthest-point (k-means++-style) from the first evicted
    chunk. Rep retention: keep the most centroid-DISTANT reps (novel within
    slot); on overflow the least distant are dropped. No temporal merge.
    """

    def __init__(self, sink=16, window=1024, m_slots=16, reps=8, route_on="resid",
                 momentum=0.9, route_chunk=256, coord=None, is_leader=True):
        super().__init__(sink, window)
        self.m_slots, self.reps, self.route_on = int(m_slots), int(reps), route_on
        self.momentum, self.route_chunk = momentum, int(route_chunk)
        self.coord, self.is_leader = coord, is_leader
        self._op_cursor = 0

    _decide = BankedRingLayer._decide  # leader logs decisions, followers replay

    def lazy_initialization(self, key_states, value_states):
        super().lazy_initialization(key_states, value_states)
        B, H, _, D = key_states.shape
        empty = lambda: torch.zeros(B, H, 0, D, dtype=self.dtype, device=self.device)
        self._sink_k, self._sink_v = empty(), empty()
        self._ring_k, self._ring_v = empty(), empty()
        self._pend_k, self._pend_v = empty(), empty()
        M = self.m_slots
        self._slot_mk = torch.zeros(B, H, M, D, dtype=torch.float32, device=self.device)
        self._slot_mv = torch.zeros_like(self._slot_mk)
        self._slot_cnt = [0] * M
        self._slot_rk = [None] * M
        self._slot_rv = [None] * M
        # leader-only routing state
        self._cent = None
        self._scores = [None] * M
        self._gmean, self._gcnt = None, 0

    def _route_decisions(self, cut_k, cut_v):
        """Leader: features -> centroid init/EMA -> assignments + rep keeps."""
        F = torch.nn.functional
        T = cut_k.shape[2]
        src = cut_v if self.route_on == "value" else cut_k
        f = src[0].permute(1, 0, 2).reshape(T, -1).float()           # [T, H*D]
        if self.route_on == "resid":
            m = f.mean(0)
            tot = self._gcnt + T
            self._gmean = m if self._gmean is None else \
                (self._gmean * self._gcnt + m * T) / tot
            self._gcnt = tot
            f = f - self._gmean
        fn = F.normalize(f, dim=-1)
        if self._cent is None:  # deterministic farthest-point init
            assert T >= self.m_slots, "first evicted chunk smaller than M"
            chosen = [fn[0]]
            sims = fn @ fn[0]
            for _ in range(self.m_slots - 1):
                nxt = int(sims.argmin())
                chosen.append(fn[nxt])
                sims = torch.maximum(sims, fn @ fn[nxt])
            self._cent = torch.stack(chosen)                         # [M, F]
        assign = (fn @ F.normalize(self._cent, dim=-1).T).argmax(-1)  # [T]
        touched = assign.unique().tolist()
        for s in touched:  # EMA on touched slots only; untouched never decay
            self._cent[s] = self.momentum * self._cent[s] + \
                (1 - self.momentum) * f[assign == s].mean(0)
        dist = 1.0 - (fn * F.normalize(self._cent, dim=-1)[assign]).sum(-1)  # [T]
        keeps = {}
        for s in touched:
            tok_idx = (assign == s).nonzero()[:, 0]
            old = self._scores[s]
            cand = dist[tok_idx] if old is None else torch.cat([old, dist[tok_idx]])
            keep = cand.topk(min(self.reps, cand.shape[0])).indices.sort().values
            keeps[s] = keep.tolist()
            self._scores[s] = cand[keep]
        return assign.tolist(), keeps

    def _route_fold(self, cut_k, cut_v):
        _, assign, keeps = self._decide(
            "route", lambda: self._route_decisions(cut_k, cut_v))
        assign = torch.tensor(assign, device=self.device)
        for s in keeps:
            tok_idx = (assign == s).nonzero()[:, 0]
            ks, vs = cut_k[:, :, tok_idx], cut_v[:, :, tok_idx]
            n_new, cnt = tok_idx.numel(), self._slot_cnt[s]
            self._slot_mk[:, :, s] = (self._slot_mk[:, :, s] * cnt + ks.float().sum(2)) / (cnt + n_new)
            self._slot_mv[:, :, s] = (self._slot_mv[:, :, s] * cnt + vs.float().sum(2)) / (cnt + n_new)
            self._slot_cnt[s] = cnt + n_new
            cand_k = ks if self._slot_rk[s] is None else torch.cat([self._slot_rk[s], ks], dim=2)
            cand_v = vs if self._slot_rv[s] is None else torch.cat([self._slot_rv[s], vs], dim=2)
            keep = torch.tensor(keeps[s], device=self.device)
            # candidate order is [old reps..., new tokens in chunk order] on
            # every layer, so the leader's keep-indices transfer directly
            self._slot_rk[s] = cand_k[:, :, keep]
            self._slot_rv[s] = cand_v[:, :, keep]

    def _materialize(self):
        parts_k, parts_v = [self._sink_k], [self._sink_v]
        for s in range(self.m_slots):
            if self._slot_cnt[s] > 0:
                parts_k.append(self._slot_mk[:, :, s : s + 1].to(self.dtype))
                parts_v.append(self._slot_mv[:, :, s : s + 1].to(self.dtype))
                parts_k.append(self._slot_rk[s])
                parts_v.append(self._slot_rv[s])
        parts_k += [self._pend_k, self._ring_k]
        parts_v += [self._pend_v, self._ring_v]
        self.keys = torch.cat([p for p in parts_k if p.numel()], dim=2)
        self.values = torch.cat([p for p in parts_v if p.numel()], dim=2)

    def n_bank_slots(self) -> int:
        return sum(1 + self._slot_rk[s].shape[2]
                   for s in range(self.m_slots) if self._slot_cnt[s] > 0)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        assert key_states.shape[0] == 1, "RoutedBankLayer is B=1 (PoC quality path)"
        ret_k = torch.cat([self.keys, key_states], dim=-2) if self.keys.numel() else key_states
        ret_v = torch.cat([self.values, value_states], dim=-2) if self.values.numel() else value_states
        self.cumulative_length += key_states.shape[-2]

        rk = torch.cat([self._ring_k, key_states], dim=2)
        rv = torch.cat([self._ring_v, value_states], dim=2)
        deficit = self.sink - self._sink_k.shape[2]
        if deficit > 0:
            take = min(deficit, rk.shape[2])
            self._sink_k = torch.cat([self._sink_k, rk[:, :, :take]], dim=2)
            self._sink_v = torch.cat([self._sink_v, rv[:, :, :take]], dim=2)
            rk, rv = rk[:, :, take:], rv[:, :, take:]
        if rk.shape[2] > self.window:
            cut = rk.shape[2] - self.window
            self._pend_k = torch.cat([self._pend_k, rk[:, :, :cut]], dim=2)
            self._pend_v = torch.cat([self._pend_v, rv[:, :, :cut]], dim=2)
            rk, rv = rk[:, :, cut:], rv[:, :, cut:]
        self._ring_k, self._ring_v = rk, rv
        if self._pend_k.shape[2] >= self.route_chunk:
            pk, pv = self._pend_k, self._pend_v
            self._pend_k, self._pend_v = pk[:, :, :0], pv[:, :, :0]
            self._route_fold(pk, pv)
        self._materialize()
        return ret_k, ret_v

    def batch_repeat_interleave(self, repeats: int):
        raise NotImplementedError("RoutedBankLayer does not support batch replication")


def build_cache(model, mode: str, args) -> DynamicCache:
    cfg = model.config.get_text_config(decoder=True)
    cache = DynamicCache(config=model.config)
    if mode in ("ring", "banked", "routed"):
        n_owned = cfg.num_hidden_layers - getattr(cfg, "num_kv_shared_layers", 0)
        ring_idx = [
            i for i in range(n_owned) if cfg.layer_types[i] == "full_attention"
        ]
        coord = [] if getattr(args, "select", "shared") == "shared" else None
        for j, i in enumerate(ring_idx):
            if mode == "ring":
                cache.layers[i] = SinkRingLayer(sink=args.sink, window=args.window)
            elif mode == "banked":
                cache.layers[i] = BankedRingLayer(
                    sink=args.sink, window=args.window, k_states=args.k_states,
                    seg=args.seg, reps=args.reps, coord=coord, is_leader=(j == 0))
            else:
                cache.layers[i] = RoutedBankLayer(
                    sink=args.sink, window=args.window,
                    m_slots=getattr(args, "m_slots", 16), reps=args.reps,
                    route_on=getattr(args, "route_on", "resid"),
                    coord=coord, is_leader=(j == 0))
    return cache


def cache_bytes(cache: DynamicCache) -> int:
    total = 0
    for layer in cache.layers:
        for t in (layer.keys, layer.values):
            if t is not None and isinstance(t, torch.Tensor):
                total += t.numel() * t.element_size()
    return total


# --------------------------------------------------------------------------- #
# Model / tokenizer loading
# --------------------------------------------------------------------------- #
def resolve_model_path(explicit: str | None) -> str:
    if explicit:
        return explicit
    for cand in MODEL_CANDIDATES:
        if os.path.isdir(cand):
            return cand
    return MODEL_CANDIDATES[-1]


def load_model(path: str, device: str = "cuda", attn: str = "sdpa"):
    """Load the text tower only (Gemma4ForCausalLM) from the multimodal
    checkpoint via key_mapping; skips vision/audio tower weights entirely."""
    from transformers.models.gemma4 import Gemma4ForCausalLM

    install_sdpa_decode_patch()
    full_cfg = AutoConfig.from_pretrained(path)
    text_cfg = full_cfg.get_text_config(decoder=True)
    model = Gemma4ForCausalLM.from_pretrained(
        path,
        config=text_cfg,
        dtype=torch.bfloat16,
        attn_implementation=attn,
        key_mapping={r"^model\.language_model": "model"},
        device_map=device,
    )
    model.eval()
    return model


def set_attn_impl(model, impl: str):
    """Switch attention implementation in place (dispatch reads config at each
    forward; mask builders read the same config object)."""
    for cfg in {id(model.config): model.config, id(model.model.config): model.model.config}.values():
        cfg._attn_implementation = impl


# --------------------------------------------------------------------------- #
# Prompt building
# --------------------------------------------------------------------------- #
def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def chat_affixes(tok):
    """Derive (prefix_ids, suffix_ids) around user content from the chat
    template, so we can token-budget the middle exactly."""
    marker = "XQZMARKERQZX"
    ids = tok.apply_chat_template(
        [{"role": "user", "content": marker}], add_generation_prompt=True
    )
    if not isinstance(ids, list):  # v5 returns a BatchEncoding
        ids = ids["input_ids"]
    marker_ids = tok(marker, add_special_tokens=False).input_ids
    pos = find_subsequence(ids, marker_ids)
    assert pos >= 0, "marker not found in chat template output"
    return ids[:pos], ids[pos + len(marker_ids) :]


def filler_ids(tok, n: int) -> list[int]:
    unit = tok(FILLER, add_special_tokens=False).input_ids
    reps = math.ceil(n / len(unit))
    return (unit * reps)[:n]


def build_plain_prompt(tok, ctx: int) -> torch.Tensor:
    """Synthetic long prompt of exactly `ctx` tokens (BOS + filler)."""
    ids = [tok.bos_token_id] + filler_ids(tok, ctx - 1)
    return torch.tensor([ids], dtype=torch.long)


def build_needle_prompt(tok, ctx: int) -> torch.Tensor:
    """Chat-formatted prompt of ~ctx tokens with the needle at ~token 200."""
    prefix, suffix = chat_affixes(tok)
    needle_ids = tok(NEEDLE, add_special_tokens=False).input_ids
    question_ids = tok("\n\n" + QUESTION, add_special_tokens=False).input_ids
    budget = ctx - len(prefix) - len(suffix) - len(needle_ids) - len(question_ids)
    assert budget > 250, f"ctx={ctx} too small for needle prompt"
    pre_needle = max(0, 200 - len(prefix))
    ids = (
        prefix
        + filler_ids(tok, pre_needle)
        + needle_ids
        + filler_ids(tok, budget - pre_needle)
        + question_ids
        + suffix
    )
    return torch.tensor([ids], dtype=torch.long)


# --------------------------------------------------------------------------- #
# Prefill / decode primitives
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def chunked_prefill(model, cache, input_ids, chunk: int, keep_last_logits: int = 1):
    """Prefill input_ids through the cache in chunks; returns logits of the last
    `keep_last_logits` positions (bf16, [1, keep, V])."""
    n = input_ids.shape[1]
    logits = None
    pos = 0
    while pos < n:
        end = min(pos + chunk, n)
        keep = keep_last_logits if end == n else 1
        out = model(
            input_ids=input_ids[:, pos:end].to(model.device),
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=keep,
        )
        logits = out.logits
        pos = end
    return logits


# SDPA decode patch: gemma-4's full-attention layers use global_head_dim=512,
# which no fused SDPA kernel on sm_89 supports with enable_gqa — torch falls
# back to the math kernel, which materializes GiB-scale fp32 buffers and is
# slower than eager. For the exact decode case (mask-free, q_len=1) attention
# is two batched GEMMs on the grouped-query layout: q [B,H,1,D] -> [B,G,H/G,D]
# (query head h uses kv head h//(H/G) — the repeat_kv/enable_gqa mapping),
# scores fp32-softmaxed like the eager path. No KV copy, no fallback.
def _grouped_gemm_decode_attention(module, query, key, value, scaling):
    B, H, _, D = query.shape
    G = key.shape[1]
    qg = query.reshape(B, G, H // G, D)
    scores = torch.matmul(qg, key.transpose(-1, -2))  # [B, G, H/G, S]
    if scaling is not None:
        scores = scores * scaling
    probs = torch.softmax(scores.float(), dim=-1).to(query.dtype)
    out = torch.matmul(probs, value)  # [B, G, H/G, D]
    return out.reshape(B, H, D).unsqueeze(1), None  # [B, q=1, H, D]


def install_sdpa_decode_patch():
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    orig = ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"]
    if getattr(orig, "_wkvm_patched", False):
        return

    def sdpa_with_wide_head_decode(module, query, key, value, attention_mask,
                                   scaling=None, **kwargs):
        if attention_mask is None and query.shape[2] == 1 and query.shape[3] > 256:
            return _grouped_gemm_decode_attention(module, query, key, value, scaling)
        return orig(module, query, key, value, attention_mask, scaling=scaling, **kwargs)

    sdpa_with_wide_head_decode._wkvm_patched = True
    ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = sdpa_with_wide_head_decode


# At q=1 decode with no padding, every cached slot is unconditionally visible
# (sink/bank/pending/ring all precede the query; a sliding layer stores exactly
# the last window-1 tokens, i.e. exactly the visible set). Passing a pre-built
# all-None mask mapping (a) skips the vmap-based per-step mask construction and
# (b) lets SDPA take its mask-free path: is_causal=False full attention over
# the cache with enable_gqa=True (no repeat_kv materialization).
DECODE_MASK_FREE = True  # --legacy-decode turns this off (baseline repro)


def _decode_mask_kwargs() -> dict:
    if DECODE_MASK_FREE:
        return {"attention_mask": {"full_attention": None, "sliding_attention": None}}
    return {}


@torch.inference_mode()
def greedy_decode(model, cache, first_token, steps, attention_mask=None):
    """Batched greedy decode. Returns (tokens [B, steps], elapsed_seconds)."""
    device = model.device
    cur = first_token.to(device)
    tokens = []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        kwargs = {}
        if attention_mask is not None:
            attention_mask = torch.cat(
                [attention_mask, torch.ones_like(attention_mask[:, :1])], dim=-1
            )
            kwargs["attention_mask"] = attention_mask
            kwargs["position_ids"] = (attention_mask.cumsum(-1) - 1)[:, -1:]
        else:
            kwargs.update(_decode_mask_kwargs())
        out = model(
            input_ids=cur,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
            **kwargs,
        )
        cur = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(cur)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return torch.cat(tokens, dim=1), elapsed


# --------------------------------------------------------------------------- #
# CUDA-graphed decode over fixed-address ring buffers
# --------------------------------------------------------------------------- #
class StaticRingLayer(DynamicLayer):
    """Fixed-address KV layer for CUDA-graphed decode.

    Preallocated [B, H, cap, D] buffers; each update() writes the new token's
    KV in place at a *device-tensor* write pointer and advances the pointer
    with captured tensor ops, so replaying the captured graph keeps mutating
    the cache correctly. The buffer's slot order becomes rotated rather than
    chronological once the ring wraps — harmless at decode because every slot
    is unconditionally visible (mask-free q=1 attention is order-invariant)
    and keys are stored post-RoPE.

    Equivalence with the dynamic layers it replaces (exact same visible set
    per step):
    - DynamicSlidingWindowLayer(window W) stores the last W-1 tokens and
      returns them + the current token: cap = W, ring over all slots.
    - SinkRingLayer(sink S, window W) returns S sink + last W tokens + the
      current token (eviction happens after the return): cap = S + W + 1,
      ring over slots [S, cap).
    """

    def __init__(self, keys, values, ptr, ring_start, cumulative_length, is_sliding):
        super().__init__()
        self.keys, self.values = keys, values
        self.cap = keys.shape[2]
        self.ring_start = int(ring_start)
        self.ring_size = self.cap - self.ring_start
        self.ptr = ptr  # LongTensor [1] on device
        self.cumulative_length = cumulative_length
        self.is_sliding = is_sliding
        self.dtype, self.device = keys.dtype, keys.device
        self.is_initialized = True

    def update(self, key_states, value_states, *args, **kwargs):
        self.keys.index_copy_(2, self.ptr, key_states)
        self.values.index_copy_(2, self.ptr, value_states)
        self.ptr.copy_(
            torch.remainder(self.ptr - self.ring_start + 1, self.ring_size)
            + self.ring_start
        )
        # Python-side counter: correct in eager use; stale under graph replay
        # (unused there — the graphed path passes explicit position_ids).
        self.cumulative_length += key_states.shape[-2]
        return self.keys, self.values

    def get_seq_length(self) -> int:
        return self.cumulative_length

    def get_max_cache_shape(self) -> int:
        return self.cap

    def batch_repeat_interleave(self, repeats: int):
        raise NotImplementedError("replicate before converting to static")


def to_static_cache(cache: DynamicCache, repeats: int = 1) -> DynamicCache:
    """Convert a prefilled ring-mode cache into StaticRingLayers in place.
    Requires every layer to be at capacity (ctx >= sink+window), which makes
    static decode token-equivalent to the dynamic layers. Frees each dynamic
    layer as it converts, so the transient memory overhead is one layer, not
    one cache. ``repeats`` broadcast-replicates a B=1 prefill directly into
    the static buffers (real per-slot copies, honest memory cost) without ever
    materializing an intermediate replicated dynamic cache."""
    for i, layer in enumerate(cache.layers):
        if isinstance(layer, SinkRingLayer):
            stored_expect = layer.sink + layer.window
            cap, start = stored_expect + 1, layer.sink
        elif isinstance(layer, DynamicSlidingWindowLayer):
            cap, start = layer.sliding_window, 0
            stored_expect = cap - 1
        else:
            raise NotImplementedError(f"layer {i}: {type(layer).__name__} not supported")
        k, v = layer.keys, layer.values
        assert k.shape[2] == stored_expect, (
            f"layer {i} stores {k.shape[2]} != {stored_expect} slots; "
            "static decode requires prefill ctx >= sink+window"
        )
        B, H, _, D = k.shape
        if repeats > 1:
            assert B == 1, "repeats>1 expects a B=1 prefill"
            B = repeats
            k, v = k.expand(B, -1, -1, -1), v.expand(B, -1, -1, -1)
        sk = torch.zeros(B, H, cap, D, dtype=k.dtype, device=k.device)
        sv = torch.zeros_like(sk)
        sk[:, :, :stored_expect].copy_(k)
        sv[:, :, :stored_expect].copy_(v)
        ptr = torch.tensor([stored_expect], dtype=torch.long, device=k.device)
        cache.layers[i] = StaticRingLayer(
            sk, sv, ptr, start, layer.cumulative_length, layer.is_sliding)
        layer.keys = layer.values = None  # free before converting the next layer
    gc.collect()
    torch.cuda.empty_cache()
    return cache


class GraphedStep:
    """One mask-free greedy decode step (forward + argmax + token feedback +
    position bump) captured in a CUDA graph over a static-ring cache.

    Static I/O: ``ids`` [B,1] (token in, next token out) and ``pos`` [1,1].
    Warmup (triton/cudnn JIT + autotune) runs eagerly on a side stream and
    therefore mutates the cache; the slots it writes and all pointers are
    snapshotted and restored before capture. Capture itself records but does
    not execute kernels, so it leaves the cache untouched."""

    def __init__(self, model, cache, batch_size, warmup_iters: int = 3):
        dev = model.device
        self.B, self.cache = batch_size, cache
        pos0 = cache.get_seq_length()
        self.ids = torch.zeros(batch_size, 1, dtype=torch.long, device=dev)
        self.pos = torch.full((1, 1), pos0, dtype=torch.long, device=dev)

        snaps = []
        for layer in cache.layers:
            p, slots = int(layer.ptr.item()), []
            for _ in range(warmup_iters):
                slots.append(p)
                p = (p - layer.ring_start + 1) % layer.ring_size + layer.ring_start
            idx = torch.tensor(sorted(set(slots)), dtype=torch.long, device=dev)
            snaps.append((layer, idx, layer.keys[:, :, idx].clone(),
                          layer.values[:, :, idx].clone(), layer.ptr.clone(),
                          layer.cumulative_length))

        with torch.inference_mode():
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(warmup_iters):
                    self._step(model)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()
            for layer, idx, ks, vs, ptr, cum in snaps:
                layer.keys[:, :, idx] = ks
                layer.values[:, :, idx] = vs
                layer.ptr.copy_(ptr)
                layer.cumulative_length = cum
            self.pos.fill_(pos0)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self._step(model)

    def _step(self, model):
        out = model(
            input_ids=self.ids,
            position_ids=self.pos,
            attention_mask={"full_attention": None, "sliding_attention": None},
            past_key_values=self.cache,
            use_cache=True,
            logits_to_keep=1,
        )
        nxt = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        self.ids.copy_(nxt)
        self.pos.add_(1)


@torch.inference_mode()
def graphed_greedy_decode(gs: GraphedStep, first_token, steps):
    """Greedy decode by graph replay; only D2D copies between steps, one sync
    at the end. Returns (tokens [B, steps] on GPU, elapsed_seconds)."""
    tokens = torch.empty(gs.B, steps, dtype=torch.long, device=gs.ids.device)
    gs.ids.copy_(first_token.to(gs.ids.device))
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in range(steps):
        gs.graph.replay()
        tokens[:, t].copy_(gs.ids[:, 0])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


@torch.inference_mode()
def _decode_argmax_gaps(model, cache, first_token, steps, force=None):
    """Greedy decode that also records the top-2 logit gap per emitted token.
    With ``force`` (a [B, steps] token matrix), inputs are teacher-forced to
    that sequence so per-step argmax can be compared across implementations
    without free-running divergence compounding. Returns (tokens, gaps)."""
    cur = first_token.to(model.device)
    toks, gaps = [], []
    for t in range(steps):
        out = model(input_ids=cur, past_key_values=cache, use_cache=True,
                    logits_to_keep=1, **_decode_mask_kwargs())
        top2 = out.logits[:, -1].float().topk(2, dim=-1)
        toks.append(top2.indices[:, :1])
        gaps.append(top2.values[:, 0] - top2.values[:, 1])
        cur = toks[-1] if force is None else force[:, t:t + 1].to(model.device)
    return torch.cat(toks, 1).cpu(), torch.stack(gaps, 1).cpu()


@torch.inference_mode()
def static_greedy_decode(model, cache, first_token, steps):
    """Eager decode over a static cache using the exact op sequence of
    GraphedStep._step — the ungraphed comparator for token-identity checks."""
    dev = model.device
    ids = first_token.to(dev)
    pos = torch.full((1, 1), cache.get_seq_length(), dtype=torch.long, device=dev)
    tokens = []
    for _ in range(steps):
        out = model(
            input_ids=ids, position_ids=pos,
            attention_mask={"full_attention": None, "sliding_attention": None},
            past_key_values=cache, use_cache=True, logits_to_keep=1)
        ids = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(ids)
        pos = pos + 1
    return torch.cat(tokens, dim=1)


def last_token_nll(model, cache, input_ids, chunk, tail=128):
    """Mean NLL of the last `tail` tokens of input_ids, teacher-forced.

    Prefills everything but the final (tail+1) tokens, then runs the final
    (tail+1)-token chunk in one forward and scores positions 1..tail of it."""
    n = input_ids.shape[1]
    assert n > tail + 1
    head, tail_ids = input_ids[:, : n - (tail + 1)], input_ids[:, n - (tail + 1) :]
    chunked_prefill(model, cache, head, chunk)
    logits = chunked_prefill(model, cache, tail_ids, chunk=tail + 1, keep_last_logits=tail + 1)
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    targets = tail_ids[:, 1:].to(logits.device)
    nll = -logprobs[:, :-1].gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item(), logits[:, -1:]


def free_cache(cache):
    del cache
    gc.collect()
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Profiling
# --------------------------------------------------------------------------- #
def run_profile(model, tok, args):
    """Decode-step breakdown at fixed B / ctx (ring mode): (a) attention ops,
    (b) other GPU compute, (c) python/launch gap (wall minus GPU busy).
    Instruments the attention interface, mask construction and cache update
    with record_function ranges; wall time comes from a separate un-profiled
    run of the same steps."""
    from torch.profiler import ProfilerActivity, profile, record_function
    import transformers.models.gemma4.modeling_gemma4 as g4
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    B, steps = args.batch_size, args.decode_tokens
    print(f"# profile: mode=ring ctx={args.ctx_per_session} B={B} steps={steps} "
          f"attn={args.attn} mask_free={DECODE_MASK_FREE}")
    warmup(model, tok, args)
    prompt = build_plain_prompt(tok, args.ctx_per_session)
    cache = build_cache(model, "ring", args)
    chunked_prefill(model, cache, prompt, args.chunk)
    for layer in cache.layers:
        layer.batch_repeat_interleave(B)
    word_ids = tok(" one two three four five six seven eight nine ten red blue",
                   add_special_tokens=False).input_ids
    first = torch.tensor([[word_ids[i % len(word_ids)]] for i in range(B)],
                         dtype=torch.long)

    # --- clean wall time (no profiler overhead) ---
    _, elapsed_clean = greedy_decode(model, cache, first, steps)

    # --- instrumented run ---
    def wrap_attn(fn):
        def inner(module, *a, **k):
            label = "ATTN_SLIDING" if getattr(module, "is_sliding", False) else "ATTN_FULL"
            with record_function(label):
                return fn(module, *a, **k)
        return inner

    def wrap(label, fn):
        def inner(*a, **k):
            with record_function(label):
                return fn(*a, **k)
        return inner

    orig_eager = g4.eager_attention_forward
    orig_sdpa = ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"]
    orig_ccm, orig_swm = g4.create_causal_mask, g4.create_sliding_window_causal_mask
    orig_update = DynamicCache.update
    g4.eager_attention_forward = wrap_attn(orig_eager)
    ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = wrap_attn(orig_sdpa)
    g4.create_causal_mask = wrap("MASK_BUILD", orig_ccm)
    g4.create_sliding_window_causal_mask = wrap("MASK_BUILD", orig_swm)
    DynamicCache.update = wrap("CACHE_UPDATE", orig_update)
    try:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            _, elapsed_prof = greedy_decode(model, cache, first, steps)
    finally:
        g4.eager_attention_forward = orig_eager
        ALL_ATTENTION_FUNCTIONS._global_mapping["sdpa"] = orig_sdpa
        g4.create_causal_mask, g4.create_sliding_window_causal_mask = orig_ccm, orig_swm
        DynamicCache.update = orig_update

    from torch.autograd import DeviceType

    LABELS = ("ATTN_FULL", "ATTN_SLIDING", "MASK_BUILD", "CACHE_UPDATE")
    ka = prof.key_averages()
    # GPU busy = sum over device-kernel events only. CPU op rows also carry
    # device time (would double-count), and each record_function label yields
    # a gpu_user_annotation row (DeviceType.CUDA) whose self time equals its
    # child kernel total — used below for attribution, excluded from busy.
    gpu_busy = sum(e.self_device_time_total for e in ka
                   if e.device_type == DeviceType.CUDA and e.key not in LABELS) / 1e6
    def dev(label):
        return sum(e.self_device_time_total for e in ka
                   if e.key == label and e.device_type == DeviceType.CUDA) / 1e6
    def cpu(label):
        return sum(e.cpu_time_total for e in ka
                   if e.key == label and e.device_type == DeviceType.CPU) / 1e6
    attn_full, attn_slide = dev("ATTN_FULL"), dev("ATTN_SLIDING")
    cache_gpu, mask_gpu = dev("CACHE_UPDATE"), dev("MASK_BUILD")
    other_gpu = gpu_busy - attn_full - attn_slide - cache_gpu - mask_gpu
    gap = elapsed_prof - gpu_busy

    ms = lambda s: s / steps * 1e3
    print(f"\n## decode-step breakdown (ring, ctx={args.ctx_per_session}, B={B}, "
          f"{steps} steps, attn={args.attn}, mask_free={DECODE_MASK_FREE})\n")
    print("| component | ms/step | % of wall |")
    print("|---|---|---|")
    rows = [
        ("wall (clean run)", elapsed_clean, ""),
        ("wall (profiled run)", elapsed_prof, None),
        ("GPU busy (sum of kernels)", gpu_busy, None),
        ("  attention: full/ring layers", attn_full, None),
        ("  attention: sliding layers", attn_slide, None),
        ("  cache update (cat/evict)", cache_gpu, None),
        ("  mask build (GPU)", mask_gpu, None),
        ("  other model compute", other_gpu, None),
        ("python/launch gap (profiled wall - GPU busy)", gap, None),
    ]
    for name, sec, _ in rows:
        print(f"| {name} | {ms(sec):.2f} | {sec / elapsed_prof * 100:.1f}% |")
    print(f"| mask build (CPU-side total) | {ms(cpu('MASK_BUILD')):.2f} | - |")
    print(f"| cache update (CPU-side total) | {ms(cpu('CACHE_UPDATE')):.2f} | - |")
    print(f"\nclean aggregate: {B * steps / elapsed_clean:.1f} tok/s "
          f"({ms(elapsed_clean):.2f} ms/step)")
    free_cache(cache)


# --------------------------------------------------------------------------- #
# Subcommands
# --------------------------------------------------------------------------- #
def warmup(model, tok, args):
    """Stabilize GPU clocks so per-run decode timings are comparable."""
    cache = build_cache(model, "full", args)
    prompt = build_plain_prompt(tok, 256)
    chunked_prefill(model, cache, prompt, args.chunk)
    greedy_decode(model, cache, prompt[:, -1:], 32)
    free_cache(cache)


def run_bench(model, tok, args):
    ctxs = args.ctxs or [900, 2048, 8192, 16384, 32768]
    ppl_ctxs = {900, 2048, 8192}
    warmup(model, tok, args)
    modes = ["full", "ring"] if args.mode == "both" else [args.mode]
    rows = []
    weights_bytes = torch.cuda.memory_allocated()
    print(f"# weights resident: {weights_bytes / 2**30:.2f} GiB")

    for mode in modes:
        for ctx in ctxs:
            prompt = build_plain_prompt(tok, ctx)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            cache = build_cache(model, mode, args)
            oom = False
            nll = float("nan")
            tps = float("nan")
            cbytes = 0
            t_prefill = float("nan")
            try:
                t0 = time.perf_counter()
                if ctx in ppl_ctxs:
                    nll, last_logits = last_token_nll(model, cache, prompt, args.chunk)
                else:
                    last_logits = chunked_prefill(model, cache, prompt, args.chunk)
                torch.cuda.synchronize()
                t_prefill = time.perf_counter() - t0
                first = last_logits[:, -1].argmax(dim=-1, keepdim=True)
                _, elapsed = greedy_decode(model, cache, first, args.decode_tokens)
                tps = args.decode_tokens / elapsed
                cbytes = cache_bytes(cache)
            except torch.OutOfMemoryError:
                oom = True
                torch.cuda.empty_cache()
            peak = torch.cuda.max_memory_allocated()
            rows.append(
                dict(
                    mode=mode, ctx=ctx, oom=oom,
                    cache_mib=cbytes / 2**20,
                    peak_gib=peak / 2**30,
                    delta_gib=(peak - base) / 2**30,
                    tps=tps, nll=nll, prefill_s=t_prefill,
                )
            )
            free_cache(cache)
            r = rows[-1]
            print(
                f"[{mode:4s}] ctx={ctx:6d} oom={oom} cache={r['cache_mib']:.1f}MiB "
                f"peak={r['peak_gib']:.2f}GiB (delta {r['delta_gib']:.2f}) "
                f"decode={r['tps']:.2f}tok/s nll_last128={r['nll']:.4f} "
                f"prefill={r['prefill_s']:.1f}s"
            )

    print("\n## Bench: full KV vs sink+ring "
          f"(sink={args.sink}, window={args.window}, decode={args.decode_tokens} tok, "
          f"chunked prefill {args.chunk})\n")
    hdr = ("| mode | ctx | cache MiB | peak GiB | peak-weights GiB | decode tok/s "
           "| NLL(last128) | prefill s |")
    print(hdr)
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["oom"]:
            print(f"| {r['mode']} | {r['ctx']} | OOM | OOM | OOM | OOM | OOM | OOM |")
        else:
            nll = "" if math.isnan(r["nll"]) else f"{r['nll']:.4f}"
            print(
                f"| {r['mode']} | {r['ctx']} | {r['cache_mib']:.1f} | {r['peak_gib']:.2f} "
                f"| {r['delta_gib']:.2f} | {r['tps']:.2f} | {nll} | {r['prefill_s']:.1f} |"
            )
    return rows


def run_needle(model, tok, args):
    modes = ["full", "ring"] if args.mode == "both" else [args.mode]
    results = []
    for ctx in [900, args.ctx]:
        for mode in modes:
            prompt = build_needle_prompt(tok, ctx)
            cache = build_cache(model, mode, args)
            logits = chunked_prefill(model, cache, prompt, args.chunk)
            first = logits[:, -1].argmax(dim=-1, keepdim=True)
            rest, _ = greedy_decode(model, cache, first, 31)
            out_ids = torch.cat([first, rest], dim=1)[0].tolist()
            eos_ids = {1, 106}  # <eos>, <end_of_turn>
            cut = next((i for i, t in enumerate(out_ids) if t in eos_ids), len(out_ids))
            out_ids = out_ids[:cut]
            text = tok.decode(out_ids, skip_special_tokens=True).strip()
            text = text.split("\n")[0][:120]
            hit = "BLUE-742" in text
            results.append(dict(ctx=ctx, mode=mode, hit=hit, text=text))
            print(f"[needle ctx={ctx:5d} {mode:4s}] recalled={hit} out: {text!r}")
            free_cache(cache)
    return results


def run_concurrency(model, tok, args):
    """Virtual-session concurrency sweep. Prefill ONE session of ctx-per-session
    tokens, then replicate its cache tensors across the batch dim to B (real
    copies via the layer API's batch_repeat_interleave — honest memory cost,
    throwaway quality) and decode batched greedy with a distinct first token per
    row. B_max = largest B that completes with >= 1 GiB device headroom."""
    steps = args.decode_tokens
    headroom = 1 << 30
    # Total bytes torch could ever use: free device memory now + what it holds,
    # clamped to the allocator cap (--mem-cap-gib keeps peak under budget on a
    # shared desktop GPU).
    avail = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved()
    avail = min(avail, int(args.mem_cap_gib * 2**30))
    print(f"# concurrency: torch-usable {avail / 2**30:.2f} GiB "
          f"(cap {args.mem_cap_gib} GiB), headroom 1 GiB, "
          f"decode {steps} tok/session, graphs={args.graphs}, attn={args.attn}")
    word_ids = tok(" one two three four five six seven eight nine ten red blue"
                   " green gold iron salt north south east west",
                   add_special_tokens=False).input_ids

    ladder = args.ladder or [8, 16, 32, 64, 128, 192, 256, 320, 384]
    modes = ["ring", "full"] if args.mode == "both" else [args.mode]
    plans = [(m, args.ctx_per_session, ladder) for m in modes]
    if args.mode == "both":
        plans.append(("full", 16384, [8, 16]))  # long-ctx collapse of full KV
        plans.append(("ring", 16384, [64]))     # must match ring @ 4096

    warmup(model, tok, args)
    tables = []
    for mode, ctx, ladder_b in plans:
        prompt = build_plain_prompt(tok, ctx)
        rows = []
        for B in ladder_b:
            gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            cache = build_cache(model, mode, args)
            row = dict(B=B, ok=False, green=False, agg=float("nan"),
                       per=float("nan"), cache_mib=float("nan"),
                       peak_gib=float("nan"), resv_gib=float("nan"))
            gs = None
            try:
                chunked_prefill(model, cache, prompt, args.chunk)
                first = torch.tensor(
                    [[word_ids[i % len(word_ids)]] for i in range(B)], dtype=torch.long)
                if args.graphs:
                    assert mode == "ring", "--graphs supports ring mode only"
                    # replicate straight into the static buffers (no transient
                    # B-wide dynamic cache)
                    cache = to_static_cache(cache, repeats=B)
                    gs = GraphedStep(model, cache, B)
                    _, elapsed = graphed_greedy_decode(gs, first, steps)
                else:
                    for layer in cache.layers:
                        layer.batch_repeat_interleave(B)
                    _, elapsed = greedy_decode(model, cache, first, steps)
                resv = torch.cuda.max_memory_reserved()
                row.update(ok=True, agg=B * steps / elapsed, per=steps / elapsed,
                           cache_mib=cache_bytes(cache) / 2**20,
                           peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                           resv_gib=resv / 2**30, green=resv <= avail - headroom)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not isinstance(exc, torch.OutOfMemoryError) and \
                        "out of memory" not in str(exc).lower():
                    raise
                gs = None  # a partially-built graph is not usable
                torch.cuda.empty_cache()
                row["resv_gib"] = torch.cuda.max_memory_reserved() / 2**30
            if gs is not None:
                del gs.graph, gs
            free_cache(cache)
            rows.append(row)
            print(f"[conc {mode:4s} ctx={ctx:5d} B={B:3d}] ok={row['ok']} "
                  f"green={row['green']} agg={row['agg']:.1f}tok/s "
                  f"per={row['per']:.2f} cache={row['cache_mib']:.0f}MiB "
                  f"reserved={row['resv_gib']:.2f}GiB")
            if not row["ok"]:
                break  # OOM: larger B is hopeless
        tables.append((mode, ctx, rows))

    for mode, ctx, rows in tables:
        green_bs = [r["B"] for r in rows if r["green"]]
        print(f"\n### concurrency {mode} @ ctx/session {ctx} — "
              f"B_max(green) = {max(green_bs) if green_bs else 0}\n")
        print("| B | cache MiB total | per-slot MiB | peak alloc GiB "
              "| peak reserved GiB | agg tok/s | per-stream tok/s | status |")
        print("|---|---|---|---|---|---|---|---|")
        for r in rows:
            if not r["ok"]:
                print(f"| {r['B']} | - | - | - | {r['resv_gib']:.2f} | - | - | OOM |")
            else:
                status = "green" if r["green"] else "over-budget"
                print(f"| {r['B']} | {r['cache_mib']:.0f} | {r['cache_mib']/r['B']:.1f} "
                      f"| {r['peak_gib']:.2f} | {r['resv_gib']:.2f} "
                      f"| {r['agg']:.1f} | {r['per']:.2f} | {status} |")
    return tables


def _needle_once(model, tok, args, mode, ctx):
    """Returns (hit, one-line output, cache MiB, bank slots on first ring layer)."""
    import copy
    a = copy.copy(args)
    prompt = build_needle_prompt(tok, ctx)
    cache = build_cache(model, mode, a)
    logits = chunked_prefill(model, cache, prompt, args.chunk)
    first = logits[:, -1].argmax(dim=-1, keepdim=True)
    rest, _ = greedy_decode(model, cache, first, 31)
    out_ids = torch.cat([first, rest], dim=1)[0].tolist()
    cut = next((i for i, t in enumerate(out_ids) if t in (1, 106)), len(out_ids))
    text = tok.decode(out_ids[:cut], skip_special_tokens=True).strip().split("\n")[0][:100]
    mib = cache_bytes(cache) / 2**20
    slots = next((l.n_bank_slots() for l in cache.layers
                  if isinstance(l, BankedRingLayer)), 0)
    free_cache(cache)
    return "BLUE-742" in text, text, mib, slots


def run_bank(model, tok, args):
    """PoC-2 driver: correctness at 900, NLL@8192, K x seg recall sweep."""
    warmup(model, tok, args)

    # --- acceptance 3: ring-exactness below window (banked == full at 900) ---
    print("\n# correctness @ ctx 900 (< sink+window: bank must be empty)")
    nlls = {}
    for mode in ["full", "ring", "banked"]:
        cache = build_cache(model, mode, args)
        nll, _ = last_token_nll(model, cache, build_plain_prompt(tok, 900), args.chunk)
        nlls[mode] = nll
        free_cache(cache)
        print(f"  NLL(last128)@900 {mode:6s} = {nll:.6f}")
    hit900, txt900, _, _ = _needle_once(model, tok, args, "banked", 900)
    print(f"  needle@900 banked recalled={hit900} out: {txt900!r}")

    # --- acceptance 5: NLL@8192 banked vs ring ---
    print("\n# NLL(last128) @ ctx 8192")
    for mode in ["full", "ring", "banked"]:
        cache = build_cache(model, mode, args)
        nll, _ = last_token_nll(model, cache, build_plain_prompt(tok, 8192), args.chunk)
        free_cache(cache)
        print(f"  NLL(last128)@8192 {mode:6s} = {nll:.4f}")

    # --- needle sweep: K_STATES x seg x ctx ---
    print(f"\n# needle sweep (reps={args.reps}: 1 mean + {args.reps - 1} representatives"
          f" per segment; sink={args.sink}, window={args.window})")
    import copy
    rows = []
    for K in [4, 16, 64]:
        for seg in [256, 512]:
            a = copy.copy(args)
            a.k_states, a.seg = K, seg
            row = dict(K=K, seg=seg)
            for ctx in [8192, 16384, 32768]:
                hit, text, mib, slots = _needle_once(model, tok, a, "banked", ctx)
                row[ctx] = hit
                row[f"mib{ctx}"] = mib
                row[f"slots{ctx}"] = slots
                print(f"[bank K={K:3d} seg={seg:3d} ctx={ctx:6d}] recalled={hit} "
                      f"cache={mib:.1f}MiB bank_slots/layer={slots} out: {text!r}")
            rows.append(row)
    # reference rows
    ring_ref = {ctx: _needle_once(model, tok, args, "ring", ctx) for ctx in [8192]}

    print("\n### PoC-2 needle recall (BLUE-742 @ ~token 200)\n")
    print("| K_STATES | seg | recall@8k | recall@16k | recall@32k "
          "| bank slots/layer @32k | cache MiB @8k/16k/32k |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['K']} | {r['seg']} | {r[8192]} | {r[16384]} | {r[32768]} "
              f"| {r['slots32768']} | {r['mib8192']:.1f} / {r['mib16384']:.1f} "
              f"/ {r['mib32768']:.1f} |")
    print(f"\nring reference @8k: recalled={ring_ref[8192][0]} "
          f"(cache {ring_ref[8192][2]:.1f} MiB); "
          f"NLL@900 full/ring/banked = {nlls['full']:.6f}/{nlls['ring']:.6f}/{nlls['banked']:.6f}")
    return rows


def run_verify(model, tok, args):
    """Correctness gates for the throughput fix (must all pass before any
    speed claim): SDPA-vs-eager greedy equivalence, NLL@900 mode-identity per
    impl, banked-mode recall under SDPA, and graphed-vs-ungraphed token
    identity over the static ring cache."""
    results = []

    def gate(name, ok, detail=""):
        results.append((name, ok))
        print(f"[gate] {'PASS' if ok else 'FAIL'}: {name} {detail}")

    warmup(model, tok, args)
    word_ids = tok(" one two three four five six seven eight nine ten red blue",
                   add_special_tokens=False).input_ids
    first4 = torch.tensor([[word_ids[i]] for i in range(4)], dtype=torch.long)
    prompt = build_plain_prompt(tok, args.ctx_per_session)

    def ring_prefill(B):
        cache = build_cache(model, "ring", args)
        chunked_prefill(model, cache, prompt, args.chunk)
        for layer in cache.layers:
            layer.batch_repeat_interleave(B)
        return cache

    # --- gate 1: NLL@900 (below sink+window: ring/banked must equal full) ---
    nll = {}
    for impl in ["eager", "sdpa"]:
        set_attn_impl(model, impl)
        for mode in ["full", "ring", "banked"]:
            cache = build_cache(model, mode, args)
            n, _ = last_token_nll(model, cache, build_plain_prompt(tok, 900), args.chunk)
            free_cache(cache)
            nll[(impl, mode)] = n
            print(f"  NLL(last128)@900 {impl:5s}/{mode:6s} = {n:.6f}")
    for impl in ["eager", "sdpa"]:
        gate(f"NLL@900 full==ring==banked ({impl})",
             nll[(impl, "full")] == nll[(impl, "ring")] == nll[(impl, "banked")])
    drift = abs(nll[("sdpa", "ring")] - nll[("eager", "ring")])
    gate("NLL@900 sdpa-vs-eager drift < 1e-3", drift < 1e-3, f"(delta={drift:.2e})")

    # --- gate 2: greedy equivalence eager vs sdpa (ring, B=4, 64 steps) ---
    # bf16 logits are quantized to ~0.125-0.25 ULP at their magnitude here, so
    # kernel-order changes can flip exact/near ties. The gate therefore accepts
    # strict equality OR: teacher-forced per-step argmax mismatches <= 2% with
    # every mismatch at a top-2 logit gap <= 0.5 (a few bf16 ULPs) in both
    # implementations — i.e. provably tie-flips, not semantic divergence.
    def tie_aware_compare(name, tokens_a, gaps_a, decode_b, decode_b_forced):
        tokens_b, _ = decode_b()
        strict = torch.equal(tokens_a, tokens_b)
        n_free = (tokens_a != tokens_b).sum().item()
        if strict:
            gate(name, True, "(strictly token-identical)")
            return
        tokens_f, gaps_f = decode_b_forced(tokens_a)
        mism = (tokens_f != tokens_a).nonzero()
        worst = max((max(gaps_a[r, t].item(), gaps_f[r, t].item())
                     for r, t in mism.tolist()), default=0.0)
        frac = len(mism) / tokens_a.numel()
        gate(name, frac <= 0.02 and worst <= 0.5,
             f"(free-run diff {n_free}/{tokens_a.numel()}; teacher-forced "
             f"argmax diff {len(mism)}/{tokens_a.numel()}, worst top-2 gap "
             f"{worst:.3f} — bf16 tie-flips)" )

    set_attn_impl(model, "eager")
    cache = ring_prefill(4)
    te, ge = _decode_argmax_gaps(model, cache, first4, 64)
    free_cache(cache)

    def sdpa_free():
        set_attn_impl(model, "sdpa")
        cache = ring_prefill(4)
        r = _decode_argmax_gaps(model, cache, first4, 64)
        free_cache(cache)
        return r

    def sdpa_forced(force):
        set_attn_impl(model, "sdpa")
        cache = ring_prefill(4)
        r = _decode_argmax_gaps(model, cache, first4, 64, force=force)
        free_cache(cache)
        return r

    tie_aware_compare(
        f"greedy tokens eager==sdpa (ring ctx={args.ctx_per_session}, B=4, 64 steps)",
        te, ge, sdpa_free, sdpa_forced)

    # --- gate 3: banked mode under sdpa (bank slots are ordinary KV) ---
    needle = {}
    for impl in ["eager", "sdpa"]:
        set_attn_impl(model, impl)
        needle[impl] = _needle_once(model, tok, args, "banked", 8192)
    gate("banked needle@8192 recalled (sdpa)", needle["sdpa"][0],
         f"out={needle['sdpa'][1]!r}")
    gate("banked greedy output eager==sdpa", needle["eager"][1] == needle["sdpa"][1])

    # --- gate 4: static ring + CUDA graph token identity (sdpa) ---
    # Static-vs-dynamic uses the same tie-aware rule (the static ring rotates
    # KV slot order, permuting fp reduction order). Graphed-vs-ungraphed runs
    # the identical kernels at identical addresses and must be token-identical.
    set_attn_impl(model, "sdpa")
    cache = ring_prefill(4)
    td, gd = _decode_argmax_gaps(model, cache, first4, 64)
    free_cache(cache)

    def static_free():
        cache = to_static_cache(ring_prefill(4))
        r = _decode_argmax_gaps(model, cache, first4, 64)
        free_cache(cache)
        return r

    def static_forced(force):
        cache = to_static_cache(ring_prefill(4))
        r = _decode_argmax_gaps(model, cache, first4, 64, force=force)
        free_cache(cache)
        return r

    tie_aware_compare("static-ring tokens == dynamic tokens (B=4, 64 steps)",
                      td, gd, static_free, static_forced)

    cache = to_static_cache(ring_prefill(4))
    static_t = static_greedy_decode(model, cache, first4, 64)
    free_cache(cache)
    cache = to_static_cache(ring_prefill(4))
    gs = GraphedStep(model, cache, 4)
    graph_t, _ = graphed_greedy_decode(gs, first4, 64)
    del gs.graph, gs
    free_cache(cache)
    n_gs = (graph_t.cpu() != static_t.cpu()).sum().item()
    gate("graphed tokens == ungraphed static tokens (must be exact)", n_gs == 0,
         f"({n_gs}/256 differ)")

    set_attn_impl(model, args.attn)
    n_fail = sum(not ok for _, ok in results)
    print(f"\n## verify: {len(results) - n_fail}/{len(results)} gates passed")
    return n_fail == 0


def run_batch(model, tok, args):
    prompts = [
        "Write one sentence about the ocean.",
        "Name three prime numbers.",
        "What is the capital of France?",
        "Give a synonym for 'happy'.",
        "What color is chlorophyll?",
        "State Newton's second law briefly.",
        "Name a famous composer.",
        "What is 12 times 12?",
    ][: args.batch_size]
    tok.padding_side = "left"
    texts = [
        tok.apply_chat_template(
            [{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False
        )
        for p in prompts
    ]
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False)
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)
    B = input_ids.shape[0]

    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    cache = build_cache(model, args.mode if args.mode != "both" else "ring", args)
    pos = (attn.cumsum(-1) - 1).clamp(min=0)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, position_ids=pos,
                    past_key_values=cache, use_cache=True, logits_to_keep=1)
    first = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    rest, elapsed = greedy_decode(model, cache, first, args.decode_tokens - 1,
                                  attention_mask=attn)
    toks = torch.cat([first, rest], dim=1)
    peak = torch.cuda.max_memory_allocated()
    agg = B * args.decode_tokens / elapsed
    print(f"\n[batch B={B} mode=ring] aggregate decode: {agg:.1f} tok/s "
          f"({args.decode_tokens} tok/seq), peak VRAM {peak / 2**30:.2f} GiB, "
          f"cache {cache_bytes(cache) / 2**20:.1f} MiB")
    for i, p in enumerate(prompts):
        text = tok.decode(toks[i], skip_special_tokens=True).strip().split("\n")[0][:80]
        print(f"  - {p!r} -> {text!r}")
    free_cache(cache)
    return agg, peak


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["bench", "needle", "batch", "concurrency", "bank",
                                    "profile", "verify"])
    ap.add_argument("--mode", choices=["full", "ring", "banked", "routed", "both"],
                    default="both")
    ap.add_argument("--attn", choices=["eager", "sdpa"], default="sdpa",
                    help="attention implementation (PoC-3 default: sdpa)")
    ap.add_argument("--legacy-decode", action="store_true",
                    help="rebuild per-step masks at decode (pre-PoC-3 behavior)")
    ap.add_argument("--graphs", action="store_true",
                    help="CUDA-graph the decode step (concurrency, ring mode)")
    ap.add_argument("--mem-cap-gib", type=float,
                    default=float(os.environ.get("WKVM_MEM_CAP_GIB", 19)),
                    help="allocator cap in GiB (leave room for the desktop)")
    ap.add_argument("--sink", type=int, default=16)
    ap.add_argument("--window", type=int, default=1024)
    ap.add_argument("--ctx", type=int, default=8192, help="needle context length")
    ap.add_argument("--chunk", type=int, default=2048, help="prefill chunk size")
    ap.add_argument("--decode-tokens", type=int, default=None,
                    help="decode steps (default: 64; 128 for concurrency)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--ctx-per-session", type=int, default=4096,
                    help="context tokens per virtual session (concurrency)")
    ap.add_argument("--ladder", type=int, nargs="*", default=None,
                    help="override concurrency batch-size ladder")
    ap.add_argument("--k-states", type=int, default=16,
                    help="bank capacity in segments (banked mode)")
    ap.add_argument("--seg", type=int, default=512,
                    help="evicted tokens per bank segment (banked mode)")
    ap.add_argument("--reps", type=int, default=8,
                    help="pseudo-KV slots per segment: 1 mean + reps-1 representatives")
    ap.add_argument("--select", choices=["shared", "per-layer"], default="shared",
                    help="representative selection: leader-layer shared indices "
                         "(default) or per-layer novelty")
    ap.add_argument("--m-slots", type=int, default=16,
                    help="routed mode: number of persistent bank slots")
    ap.add_argument("--route-on", choices=["key", "resid", "value"], default="resid",
                    help="routed mode: routing feature")
    ap.add_argument("--ctxs", type=int, nargs="*", default=None,
                    help="override bench context lengths")
    ap.add_argument("--model-path", default=None)
    args = ap.parse_args()
    if args.decode_tokens is None:
        args.decode_tokens = {"concurrency": 128, "profile": 32}.get(args.cmd, 64)
    if args.legacy_decode:
        global DECODE_MASK_FREE
        DECODE_MASK_FREE = False

    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, args.mem_cap_gib * 2**30 / total))

    path = resolve_model_path(args.model_path)
    print(f"# loading {path} (text tower only, {args.attn}, bf16, "
          f"mem cap {args.mem_cap_gib} GiB)")
    t0 = time.perf_counter()
    model = load_model(path, attn=args.attn)
    tok = AutoTokenizer.from_pretrained(path)
    print(f"# loaded in {time.perf_counter() - t0:.1f}s; "
          f"weights {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    cfg = model.config.get_text_config(decoder=True)
    n_owned = cfg.num_hidden_layers - getattr(cfg, "num_kv_shared_layers", 0)
    ring_idx = [i for i in range(n_owned) if cfg.layer_types[i] == "full_attention"]
    print(f"# growing-KV (ring-capped) layers: {ring_idx}; "
          f"owned layers: {n_owned}/{cfg.num_hidden_layers}")

    if args.cmd == "bench":
        run_bench(model, tok, args)
    elif args.cmd == "needle":
        run_needle(model, tok, args)
    elif args.cmd == "batch":
        run_batch(model, tok, args)
    elif args.cmd == "concurrency":
        run_concurrency(model, tok, args)
    elif args.cmd == "bank":
        run_bank(model, tok, args)
    elif args.cmd == "profile":
        run_profile(model, tok, args)
    elif args.cmd == "verify":
        ok = run_verify(model, tok, args)
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
