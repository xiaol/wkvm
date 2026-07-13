#!/usr/bin/env bash
set -euo pipefail

bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$bundle_dir"

sha256sum -c SHA256SUMS

if [[ "${1:-}" != "--external" ]]; then
    exit 0
fi

repo_root=$(cd "$bundle_dir/../../.." && pwd)
while IFS=$'\t' read -r expected_sha expected_size relative_path; do
    target="$repo_root/$relative_path"
    actual_sha=$(sha256sum -- "$target" | cut -d' ' -f1)
    actual_size=$(stat -c '%s' -- "$target")
    if [[ "$actual_sha" != "$expected_sha" || "$actual_size" != "$expected_size" ]]; then
        printf 'promoted-file mismatch: %s\n' "$relative_path" >&2
        exit 1
    fi
done < provenance/promoted_files.tsv

printf 'promoted source and artifact manifest: OK\n'
