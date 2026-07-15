"""Stable source-tree and model-file identities for benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Iterable


SOURCE_IDENTITY_SCHEMA = "wkvm.git_worktree_identity.sha256.v1"
MODEL_IDENTITY_SCHEMA = "wkvm.model_checkpoint_identity.sha256.v1"
SOURCE_EXCLUDED_PATH_PATTERNS = (
    "experiments/results/**",
    "**/__pycache__/**",
    ".pytest_cache/**",
    "**/*.egg-info/**",
    ".venv/**",
    "build/**",
    "dist/**",
)
MODEL_EXCLUDED_PATH_PATTERNS = (".cache/**",)
SOURCE_IDENTITY_SCOPE = (
    "git tracked and all untracked worktree files excluding declared generated artifacts"
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_bytes(root: Path, *arguments: str) -> bytes:
    return subprocess.check_output(
        ["git", *arguments],
        cwd=root,
        stderr=subprocess.DEVNULL,
    )


def _git_paths(root: Path, *arguments: str) -> set[bytes]:
    return {
        value
        for value in _git_bytes(root, "ls-files", "-z", *arguments).split(b"\0")
        if value
    }


def _worktree_manifest(
    root: Path,
    excluded_paths: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    tracked = _git_paths(root, "--cached")
    untracked = _git_paths(root, "--others")
    tracked = {
        path
        for path in tracked
        if not _excluded_worktree_path(os.fsdecode(path), excluded_paths)
    }
    untracked = {
        path
        for path in untracked
        if not _excluded_worktree_path(os.fsdecode(path), excluded_paths)
    }
    entries: list[dict[str, Any]] = []
    for encoded_path in sorted(tracked | untracked):
        relative_path = os.fsdecode(encoded_path)
        path = root / relative_path
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            entries.append(
                {
                    "path": relative_path,
                    "tracked": encoded_path in tracked,
                    "kind": "missing",
                    "mode": None,
                    "size_bytes": None,
                    "sha256": None,
                }
            )
            continue
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            content = os.fsencode(os.readlink(path))
            kind = "symlink"
            size_bytes = len(content)
            digest = _sha256_bytes(content)
        elif path.is_file():
            kind = "file"
            size_bytes = metadata.st_size
            digest = _sha256_file(path)
        else:
            kind = "special"
            size_bytes = metadata.st_size
            digest = None
        entries.append(
            {
                "path": relative_path,
                "tracked": encoded_path in tracked,
                "kind": kind,
                "mode": mode,
                "size_bytes": size_bytes,
                "sha256": digest,
            }
        )
    return entries, len(tracked), len(untracked)


def _excluded_worktree_path(relative_path: str, excluded_paths: set[str]) -> bool:
    normalized = relative_path.replace(os.sep, "/")
    parts = normalized.split("/")
    generated = (
        normalized.startswith("experiments/results/")
        or "__pycache__" in parts
        or normalized.startswith(".pytest_cache/")
        or any(part.endswith(".egg-info") for part in parts)
        or parts[0] in {".venv", "build", "dist"}
    )
    return generated or normalized in excluded_paths


def _relative_excluded_paths(
    root: Path,
    excluded_paths: Iterable[str | Path],
) -> set[str]:
    relative_paths: set[str] = set()
    for value in excluded_paths:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            relative = path.resolve().relative_to(root)
        except (OSError, ValueError):
            continue
        normalized = relative.as_posix()
        if not _excluded_worktree_path(normalized, set()):
            relative_paths.add(normalized)
    return relative_paths


def source_worktree_identity(
    root: Path,
    *,
    excluded_paths: Iterable[str | Path] = (),
) -> dict[str, Any]:
    root = Path(root).resolve()
    relative_exclusions = _relative_excluded_paths(root, excluded_paths)
    pathspecs = [
        ".",
        *(
            f":(exclude,glob){pattern}"
            for pattern in SOURCE_EXCLUDED_PATH_PATTERNS
        ),
        *(f":(exclude,literal){path}" for path in sorted(relative_exclusions)),
    ]
    try:
        commit = _git_bytes(root, "rev-parse", "HEAD").decode().strip()
        head_tree = _git_bytes(root, "rev-parse", "HEAD^{tree}").decode().strip()
        status = _git_bytes(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
            "--",
            *pathspecs,
        )
        tracked_diff = _git_bytes(
            root,
            "diff",
            "--binary",
            "HEAD",
            "--",
            *pathspecs,
        )
        manifest, tracked_count, untracked_count = _worktree_manifest(
            root,
            relative_exclusions,
        )
        manifest_sha256 = _canonical_sha256(manifest)
        identity_fields = {
            "git_commit": commit,
            "git_head_tree": head_tree,
            "git_status_sha256": _sha256_bytes(status),
            "git_tracked_diff_sha256": _sha256_bytes(tracked_diff),
            "worktree_manifest_sha256": manifest_sha256,
        }
        return {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "repo_root": str(root),
            "scope": SOURCE_IDENTITY_SCOPE,
            "excluded_path_patterns": list(SOURCE_EXCLUDED_PATH_PATTERNS),
            "excluded_paths": sorted(relative_exclusions),
            "git_worktree_dirty": bool(status) or untracked_count > 0,
            "tracked_file_count": tracked_count,
            "untracked_file_count": untracked_count,
            "worktree_file_count": len(manifest),
            **identity_fields,
            "identity_sha256": _canonical_sha256(identity_fields),
            "error": None,
        }
    except Exception as exc:
        return {
            "schema": SOURCE_IDENTITY_SCHEMA,
            "repo_root": str(root),
            "identity_sha256": None,
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }


def model_checkpoint_identity(model_path: str | Path) -> dict[str, Any]:
    requested_path = Path(model_path).expanduser()
    try:
        root = requested_path.resolve(strict=True)
        if root.is_file():
            files = [root]
            manifest_root = root.parent
        elif root.is_dir():
            files = sorted(
                (
                    path
                    for path in root.rglob("*")
                    if path.is_file()
                    and ".cache" not in path.relative_to(root).parts
                ),
                key=lambda path: path.relative_to(root).as_posix(),
            )
            manifest_root = root
        else:
            raise ValueError("model path is neither a regular file nor a directory")
        manifest = [
            {
                "path": path.relative_to(manifest_root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
            for path in files
        ]
        if not manifest:
            raise ValueError("model path contains no regular files")
        return {
            "schema": MODEL_IDENTITY_SCHEMA,
            "model_root": str(root),
            "excluded_path_patterns": list(MODEL_EXCLUDED_PATH_PATTERNS),
            "file_count": len(manifest),
            "total_bytes": sum(entry["size_bytes"] for entry in manifest),
            "files": manifest,
            "manifest_sha256": _canonical_sha256(manifest),
            "error": None,
        }
    except Exception as exc:
        return {
            "schema": MODEL_IDENTITY_SCHEMA,
            "model_root": str(requested_path),
            "file_count": 0,
            "total_bytes": 0,
            "files": [],
            "manifest_sha256": None,
            "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}",
        }
