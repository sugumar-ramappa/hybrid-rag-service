#!/usr/bin/env bash
#
# Fetch the evaluation corpus: Kubernetes documentation.
#
#   ./scripts/fetch_corpus.sh                    # concepts + tasks (default)
#   ./scripts/fetch_corpus.sh concepts           # one section
#   ./scripts/fetch_corpus.sh concepts tasks reference
#
# The corpus is NOT committed - it is several hundred files that belong to the
# Kubernetes project, not to this repo. documents/ is gitignored; anyone can
# reproduce the exact corpus by running this.
#
set -euo pipefail

REPO="https://github.com/kubernetes/website.git"
DEST="${DOCUMENTS_DIR:-./documents}"
SECTIONS=("${@:-concepts tasks}")

# Word-split the default when no arguments were given.
if [ "$#" -eq 0 ]; then
    SECTIONS=(concepts tasks)
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "Cloning $REPO (blobless, sparse)..."
# --filter=blob:none downloads file contents lazily, --sparse checks out
# nothing until we ask. Together they avoid pulling a large site repo to get
# a few hundred markdown files.
git clone --depth 1 --filter=blob:none --sparse "$REPO" "$tmp/website" --quiet

paths=()
for section in "${SECTIONS[@]}"; do
    paths+=("content/en/docs/$section")
done

echo "Checking out: ${paths[*]}"
git -C "$tmp/website" sparse-checkout set "${paths[@]}"

mkdir -p "$DEST"
for section in "${SECTIONS[@]}"; do
    src="$tmp/website/content/en/docs/$section"
    if [ ! -d "$src" ]; then
        echo "  skipped '$section' - not found in the repo" >&2
        continue
    fi
    echo "  copying $section"
    cp -R "$src" "$DEST/"
done

count="$(find "$DEST" -name '*.md' -type f | wc -l | tr -d ' ')"
echo
echo "Done. $count markdown files in $DEST"
echo "Next: python -m scripts.ingest"
