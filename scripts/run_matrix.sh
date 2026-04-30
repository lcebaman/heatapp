#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./mpi_probe}"
RANKS="${RANKS:-2 4 8}"
MIN_BYTES="${MIN_BYTES:-8}"
MAX_BYTES="${MAX_BYTES:-1048576}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-20}"
OUT_DIR="${OUT_DIR:-results}"
MPIEXEC="${MPIEXEC:-mpirun}"

mkdir -p "$OUT_DIR"

for np in $RANKS; do
  stamp="$(date +%Y%m%d-%H%M%S)"
  out="$OUT_DIR/mpi-probe-np${np}-${stamp}.jsonl"
  echo "running np=$np -> $out" >&2
  "$MPIEXEC" -np "$np" "$BIN" \
    --min-bytes "$MIN_BYTES" \
    --max-bytes "$MAX_BYTES" \
    --iters "$ITERS" \
    --warmup "$WARMUP" | tee "$out"
done
