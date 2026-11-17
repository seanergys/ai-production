#!/usr/bin/env bash
# Build the release archive the deploy ships: this repo at <sha>, plus upstream modelzoo at
# the commit pinned by MODELZOO_REF in the Makefile, unpacked under modelzoo/.
#
# Why modelzoo travels with the release: the node used to copy a modelzoo snapshot from its
# state directory into every release. That snapshot was seeded once and never refreshed, so
# serving ran months-old model code while training ran the pinned one. Models whose saved
# artefact imports modelzoo classes (e.g. DurationPredictionBull's _DurationEstimator) then
# failed to load. lxp_release.sh only falls back to that snapshot when the release has no
# modelzoo of its own, so shipping it here is all it takes — and a rollback restores the
# modelzoo that release was deployed with.
#
# usage: build_release_archive.sh <out.tar.gz> <sha>
#   MODELZOO_SOURCE  clone URL/path for modelzoo (default: this GitLab, via GITLAB_TOKEN)
set -euo pipefail

OUT="${1:?usage: $0 <out.tar.gz> <sha>}"
SHA="${2:?usage: $0 <out.tar.gz> <sha>}"
ROOT="$(git rev-parse --show-toplevel)"

REF="$(sed -n 's/^MODELZOO_REF[[:space:]]*?=[[:space:]]*//p' "$ROOT/Makefile" | head -n1 | tr -d '[:space:]')"
if [ "${#REF}" -ne 40 ] || [ -n "$(printf '%s' "$REF" | tr -d '0-9a-f')" ]; then
    echo "error: MODELZOO_REF in Makefile must be a full 40-character commit id, got '$REF'" >&2
    exit 1
fi

SOURCE="${MODELZOO_SOURCE:-}"
if [ -z "$SOURCE" ]; then
    SOURCE="https://oauth2:${GITLAB_TOKEN:?GITLAB_TOKEN is required to fetch modelzoo}@${CI_SERVER_HOST:?CI_SERVER_HOST is required}/software/modelzoo.git"
fi

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

# Blobless: only the pinned commit's files are downloaded, by `git archive` below.
git clone --quiet --filter=blob:none --no-checkout "$SOURCE" "$work/modelzoo"
git -C "$work/modelzoo" cat-file -e "${REF}^{commit}" 2>/dev/null || {
    echo "error: modelzoo has no commit $REF" >&2
    exit 1
}

git -C "$ROOT" archive --format=tar -o "$work/release.tar" "$SHA"
# data/ is a dataset download cache committed upstream (~0.5 GB of M100 parquet), read only
# by M100Dataset as ./data relative to the caller's cwd. No served or trained model here
# needs it, and it would make every release — of which the node keeps several — 30x larger.
git -C "$work/modelzoo" archive --format=tar --prefix=modelzoo/ -o "$work/modelzoo.tar" \
    "$REF" -- . ':(exclude)data'
tar --concatenate -f "$work/release.tar" "$work/modelzoo.tar"

# Record which modelzoo this release carries, for whoever inspects the node later.
mkdir -p "$work/meta/modelzoo"
printf '%s\n' "$REF" > "$work/meta/modelzoo/.examlops-modelzoo-ref"
tar -rf "$work/release.tar" -C "$work/meta" modelzoo/.examlops-modelzoo-ref

gzip -n -c "$work/release.tar" > "$OUT"
echo "release archive: ${SHA} + modelzoo@${REF}"
