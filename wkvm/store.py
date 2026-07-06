"""StateStore: durable, named, versioned, forkable, MUTABLE state handles.

This is the layer that makes wkvm's states *objects* rather than caches. A
handle names an immutable snapshot of one session's complete recurrent state
(every family in the model's ``ModelStateSpec``) plus enough metadata to
resume, fork, or mutate it. The deliberate invariant violation: a mutated
state is NOT ``f(token-prefix)`` for any prefix — provenance is recorded as
``(parent, rule, params)`` lineage instead, which is exactly what
prefix-hash/radix cache indexes cannot represent (docs/ANGLE.md §5).

Handles are ``name@version`` strings. Versions are append-only per name;
records are immutable once written (fork/mutate/save create new versions).

Tiers:
- HOT:  arena slot in the ``RWKV7StateBank`` (owned by the engine, not here).
- WARM: pinned host tensors held by this store.
- COLD: one safetensors file per handle + a JSON index, rewritten atomically,
  under ``store_dir``. A fresh process can rebuild the store from the index
  alone — restart persistence needs nothing in memory.

Concurrency: synchronous, single-threaded-engine assumption throughout (the
HTTP layer serialises access). D2H/H2D copies synchronize before returning.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import torch

from wkvm.runner.state import RWKV7StateBank

MutationRule = Callable[[dict[str, torch.Tensor], dict], dict[str, torch.Tensor]]


@dataclass(frozen=True)
class StateRecord:
    """Metadata for one immutable state snapshot."""

    name: str
    version: int
    fingerprint: str
    # Tokens whose state is baked into the snapshot ...
    num_computed_tokens: int
    # ... and the full known token list (may exceed num_computed_tokens by a
    # trailing sampled-but-not-yet-fed token). Resume rebuilds a Request from
    # exactly these two numbers; the gap >= 1 keeps it schedulable.
    token_ids: tuple[int, ...]
    parent: str | None = None
    rule: str | None = None
    rule_params: dict = field(default_factory=dict)

    @property
    def handle(self) -> str:
        return f"{self.name}@{self.version}"


class FingerprintMismatch(RuntimeError):
    pass


class StateStore:
    def __init__(
        self,
        bank: RWKV7StateBank,
        store_dir: str | Path,
    ) -> None:
        self.bank = bank
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.fingerprint = self._fingerprint(bank)
        self._warm: dict[str, dict[str, torch.Tensor]] = {}
        self._meta: dict[str, StateRecord] = {}
        self._cold: set[str] = set()
        self._rules: dict[str, MutationRule] = {}
        self.register_rule("decay", _rule_decay)
        self.register_rule("merge", self._rule_merge)
        self._load_index()

    # -- identity ----------------------------------------------------------

    @staticmethod
    def _fingerprint(bank: RWKV7StateBank) -> str:
        l = bank.layout
        key = f"rwkv7:L{l.n_layer}:wkv{tuple(l.wkv_shape)}:shift{tuple(l.shift_shape)}:{l.dtype}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def _next_version(self, name: str) -> int:
        versions = [r.version for r in self._meta.values() if r.name == name]
        return max(versions, default=-1) + 1

    def get(self, handle: str) -> StateRecord:
        try:
            return self._meta[handle]
        except KeyError:
            raise KeyError(f"unknown state handle {handle!r}") from None

    def list(self) -> list[StateRecord]:
        return sorted(self._meta.values(), key=lambda r: (r.name, r.version))

    # -- snapshot / restore --------------------------------------------------

    def save(
        self,
        name: str,
        slots: dict[str, int],
        num_computed_tokens: int,
        token_ids: list[int],
        parent: str | None = None,
        rule: str | None = None,
        rule_params: dict | None = None,
    ) -> str:
        """Snapshot HOT slot state into a new WARM record. One D2H per family."""
        record = StateRecord(
            name=name,
            version=self._next_version(name),
            fingerprint=self.fingerprint,
            num_computed_tokens=num_computed_tokens,
            token_ids=tuple(token_ids),
            parent=parent,
            rule=rule,
            rule_params=rule_params or {},
        )
        tensors = {
            "wkv": self._to_host(self.bank.wkv[:, slots["wkv"]]),
            "shift": self._to_host(self.bank.shift[:, :, slots["shift"]]),
        }
        torch.cuda.synchronize()
        self._warm[record.handle] = tensors
        self._meta[record.handle] = record
        self._write_index()
        return record.handle

    def load(self, handle: str, slots: dict[str, int]) -> StateRecord:
        """Restore a record into already-allocated arena slots.

        The sub-100ms path: WARM tensors are pinned, so this is one async H2D
        per family into the bank's slot views, then one synchronize."""
        record = self.get(handle)
        if record.fingerprint != self.fingerprint:
            raise FingerprintMismatch(
                f"{handle}: saved for model {record.fingerprint}, "
                f"engine is {self.fingerprint}"
            )
        tensors = self._tensors(handle)
        self.bank.wkv[:, slots["wkv"]].copy_(tensors["wkv"], non_blocking=True)
        self.bank.shift[:, :, slots["shift"]].copy_(tensors["shift"], non_blocking=True)
        torch.cuda.synchronize()
        return record

    # -- lineage ---------------------------------------------------------------

    def fork(self, handle: str, new_name: str) -> str:
        """New name sharing the parent's bytes (records are immutable, so
        sharing WARM tensors is safe; a fork touches no state data)."""
        parent = self.get(handle)
        record = StateRecord(
            name=new_name,
            version=self._next_version(new_name),
            fingerprint=parent.fingerprint,
            num_computed_tokens=parent.num_computed_tokens,
            token_ids=parent.token_ids,
            parent=handle,
        )
        if handle in self._warm:
            self._warm[record.handle] = self._warm[handle]
        else:  # COLD parent: the child references the same tensors on load
            self._warm[record.handle] = self._tensors(handle)
        self._meta[record.handle] = record
        self._write_index()
        return record.handle

    def mutate(self, handle: str, rule: str, params: dict | None = None) -> str:
        """Apply a registered state-update rule; returns a new version with
        full provenance. The result is intentionally not reproducible from
        any token prefix — that is the point."""
        params = params or {}
        fn = self._rules[rule]
        parent = self.get(handle)
        src = self._tensors(handle)
        out = fn({k: v.clone() for k, v in src.items()}, params)
        record = StateRecord(
            name=parent.name,
            version=self._next_version(parent.name),
            fingerprint=parent.fingerprint,
            num_computed_tokens=parent.num_computed_tokens,
            token_ids=parent.token_ids,
            parent=handle,
            rule=rule,
            rule_params=params,
        )
        self._warm[record.handle] = {k: self._pin(v) for k, v in out.items()}
        self._meta[record.handle] = record
        self._write_index()
        return record.handle

    def register_rule(self, name: str, fn: MutationRule) -> None:
        self._rules[name] = fn

    # -- tiering -----------------------------------------------------------------

    def persist(self, handle: str) -> Path:
        """WARM -> COLD (safetensors + index). Idempotent."""
        from safetensors.torch import save_file

        record = self.get(handle)
        path = self._cold_path(handle)
        if handle not in self._cold:
            tensors = self._tensors(handle)
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))
            self._cold.add(handle)
            self._write_index()
        return path

    def evict(self, handle: str) -> None:
        """Drop the WARM copy (persisting first if needed)."""
        self.persist(handle)
        self._warm.pop(handle, None)

    def delete(self, handle: str) -> None:
        self._warm.pop(handle, None)
        self._meta.pop(handle, None)
        if handle in self._cold:
            self._cold.discard(handle)
            self._cold_path(handle).unlink(missing_ok=True)
        self._write_index()

    # -- internals -------------------------------------------------------------------

    def _tensors(self, handle: str) -> dict[str, torch.Tensor]:
        if handle in self._warm:
            return self._warm[handle]
        if handle in self._cold:
            from safetensors.torch import load_file

            tensors = {
                k: self._pin(v) for k, v in load_file(str(self._cold_path(handle))).items()
            }
            self._warm[handle] = tensors  # promote
            return tensors
        raise KeyError(f"{handle}: metadata present but no WARM or COLD data")

    def _to_host(self, view: torch.Tensor) -> torch.Tensor:
        host = torch.empty(view.shape, dtype=view.dtype, pin_memory=True)
        host.copy_(view, non_blocking=True)
        return host

    @staticmethod
    def _pin(t: torch.Tensor) -> torch.Tensor:
        return t if t.is_pinned() else t.pin_memory()

    def _cold_path(self, handle: str) -> Path:
        return self.store_dir / f"{handle.replace('@', '_v')}.safetensors"

    def _rule_merge(
        self, tensors: dict[str, torch.Tensor], params: dict
    ) -> dict[str, torch.Tensor]:
        other = self._tensors(params["other"])
        w = float(params.get("weight", 0.5))
        return {
            k: ((1.0 - w) * tensors[k].float() + w * other[k].float()).to(tensors[k].dtype)
            for k in tensors
        }

    # -- index (restart persistence) -----------------------------------------------

    def _index_path(self) -> Path:
        return self.store_dir / "index.json"

    def _write_index(self) -> None:
        data = {
            h: {**asdict(r), "token_ids": list(r.token_ids), "cold": h in self._cold}
            for h, r in self._meta.items()
        }
        tmp = self._index_path().with_suffix(".tmp")
        tmp.write_text(json.dumps(data))
        os.replace(tmp, self._index_path())

    def _load_index(self) -> None:
        if not self._index_path().exists():
            return
        for handle, entry in json.loads(self._index_path().read_text()).items():
            cold = entry.pop("cold", False)
            entry["token_ids"] = tuple(entry["token_ids"])
            entry["rule_params"] = entry.get("rule_params") or {}
            self._meta[handle] = StateRecord(**entry)
            if cold:
                self._cold.add(handle)
            # WARM-only records from a dead process are unrecoverable; keep
            # the metadata (lineage) but loads will fail until re-persisted.


def _rule_decay(tensors: dict[str, torch.Tensor], params: dict) -> dict[str, torch.Tensor]:
    """Scale the associative wkv memory toward zero; token-shift untouched.

    The simplest useful mutation: softly forget, keeping recency channels."""
    alpha = float(params.get("alpha", 0.9))
    out = dict(tensors)
    out["wkv"] = (tensors["wkv"].float() * alpha).to(tensors["wkv"].dtype)
    return out
