#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$bundle_dir"

sha256sum -c SHA256SUMS

if [[ "${1:-}" != "--external" ]]; then
    exit 0
fi

model_root=$(jq -r '.model_root' provenance/model_identity.json)
(
    cd "$model_root"
    sha256sum -c "$bundle_dir/provenance/model_files.sha256"
)

repo_root=$(cd "$bundle_dir/../../.." && pwd)
while IFS=$'\t' read -r expected_sha expected_size relative_path; do
    target="$repo_root/$relative_path"
    actual_sha=$(sha256sum -- "$target" | cut -d' ' -f1)
    actual_size=$(stat -c '%s' -- "$target")
    if [[ "$actual_sha" != "$expected_sha" || "$actual_size" != "$expected_size" ]]; then
        printf 'source mismatch: %s\n' "$relative_path" >&2
        exit 1
    fi
done < provenance/worktree_files.tsv

printf 'external checkpoint and worktree manifests: OK\n'
