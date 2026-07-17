#!/usr/bin/env bash
set -euo pipefail

output_path=${1:-/tmp/wkvm-hf-alice-11.txt}
expected_sha256=f17aa0bf7466424a8b357b688678666bad7a0148963ef349016a3098faa6bd1e
parquet_url=https://huggingface.co/datasets/common-pile/project_gutenberg/resolve/d0bf09a2c2f6f73952733d7a1fe9a34b1cb4348c/default/partial-train/0000.parquet

mkdir -p "$(dirname "$output_path")"
temporary_path=$(mktemp "${output_path}.tmp.XXXXXX")
trap 'rm -f "$temporary_path"' EXIT

hf datasets sql \
  "SELECT text FROM read_parquet('$parquet_url') WHERE id = '11'" \
  --format json \
  | jq -je \
    'if length == 1 and (.[0].text | type) == "string" then .[0].text else error("expected one text row") end' \
  >"$temporary_path"

printf '%s  %s\n' "$expected_sha256" "$temporary_path" | sha256sum -c -
mv "$temporary_path" "$output_path"
trap - EXIT
printf 'wrote %s\n' "$output_path"
