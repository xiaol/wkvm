"""Minimal /v1/states + /v1/generate HTTP layer (stdlib only).

This is the API *shape* of the Durable State design, not a production
server: one engine, one lock, requests served synchronously. The point it
demonstrates is that every session-lifecycle operation — create, resume,
fork, mutate, persist — is a handle operation with O(state-size) cost.

  POST   /v1/states                {"prompt_ids": [...], "name": "..."}
  GET    /v1/states                list records
  GET    /v1/states/<handle>       metadata
  POST   /v1/states/<handle>/fork  {"name": "..."}
  POST   /v1/states/<handle>/mutate {"rule": "...", "params": {...}}
  POST   /v1/states/<handle>/persist
  DELETE /v1/states/<handle>
  POST   /v1/generate              {"state": handle | "prompt_ids": [...],
                                    "max_tokens": n, "temperature": t,
                                    "seed": s, "save_as": name?}

Run: python -m wkvm.server --model <path> --store <dir> [--port 8000]
"""

from __future__ import annotations

import json
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


def build_app(engine):
    """Returns a request-handler class bound to one engine + one big lock."""
    lock = threading.Lock()

    def run_request(req) -> list[int]:
        while not req.status.is_finished:
            engine.step()
        return list(req.output_token_ids)

    class Handler(BaseHTTPRequestHandler):
        def _json(self, code: int, payload) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _body(self) -> dict:
            n = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(n) or b"{}")

        def log_message(self, *a) -> None:  # quiet
            pass

        def do_GET(self) -> None:
            parts = urlparse(self.path).path.strip("/").split("/")
            with lock:
                if parts[:2] == ["v1", "states"] and len(parts) == 2:
                    self._json(200, [asdict(r) for r in engine.store.list()])
                elif parts[:2] == ["v1", "states"] and len(parts) == 3:
                    try:
                        self._json(200, asdict(engine.store.get(parts[2])))
                    except KeyError:
                        self._json(404, {"error": f"unknown handle {parts[2]}"})
                else:
                    self._json(404, {"error": "not found"})

        def do_DELETE(self) -> None:
            parts = urlparse(self.path).path.strip("/").split("/")
            if parts[:2] == ["v1", "states"] and len(parts) == 3:
                with lock:
                    engine.store.delete(parts[2])
                self._json(200, {"deleted": parts[2]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self) -> None:
            from wkvm.core.request import Request
            from wkvm.runner.sampling import SamplingParams

            parts = urlparse(self.path).path.strip("/").split("/")
            body = self._body()
            try:
                with lock:
                    if parts == ["v1", "states"]:
                        req = Request(
                            prompt_token_ids=list(body["prompt_ids"]),
                            max_new_tokens=1,
                        )
                        engine.add_request(req)
                        engine.save_on_finish(req.req_id, body.get("name", "state"))
                        run_request(req)
                        self._json(200, {"handle": engine._finish_handles[req.req_id]})
                    elif parts[:2] == ["v1", "states"] and parts[3:] == ["fork"]:
                        self._json(
                            200,
                            {"handle": engine.store.fork(parts[2], body["name"])},
                        )
                    elif parts[:2] == ["v1", "states"] and parts[3:] == ["mutate"]:
                        h = engine.store.mutate(
                            parts[2], body["rule"], body.get("params", {})
                        )
                        self._json(200, {"handle": h})
                    elif parts[:2] == ["v1", "states"] and parts[3:] == ["persist"]:
                        self._json(200, {"path": str(engine.store.persist(parts[2]))})
                    elif parts == ["v1", "generate"]:
                        params = SamplingParams(
                            temperature=float(body.get("temperature", 0.0)),
                            seed=body.get("seed"),
                        )
                        if "state" in body:
                            req = engine.submit_from_handle(
                                body["state"],
                                suffix_tokens=body.get("prompt_ids"),
                                max_new_tokens=int(body.get("max_tokens", 128)),
                                params=params,
                            )
                        else:
                            req = Request(
                                prompt_token_ids=list(body["prompt_ids"]),
                                max_new_tokens=int(body.get("max_tokens", 128)),
                            )
                            engine.add_request(req, params)
                        if body.get("save_as"):
                            engine.save_on_finish(req.req_id, body["save_as"])
                        tokens = run_request(req)
                        out = {"tokens": tokens}
                        if body.get("save_as"):
                            out["handle"] = engine._finish_handles.get(req.req_id)
                        self._json(200, out)
                    else:
                        self._json(404, {"error": "not found"})
            except (KeyError, ValueError) as e:
                self._json(400, {"error": str(e)})

    return Handler


def main() -> None:
    import argparse

    from wkvm.engine import Engine

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--store", required=True)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    engine = Engine.from_pretrained(args.model, num_slots=args.slots)
    engine.attach_store(args.store)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), build_app(engine))
    print(f"wkvm /v1/states serving on 127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
