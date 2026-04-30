#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./mpi_probe}"
LAUNCHER="${LAUNCHER:-slurm}"
NODES="${NODES:-1 2}"
RANKS_PER_NODE="${RANKS_PER_NODE:-1 2 4 8}"
MIN_BYTES="${MIN_BYTES:-8}"
MAX_BYTES="${MAX_BYTES:-1048576}"
ALLTOALL_MAX_BYTES="${ALLTOALL_MAX_BYTES:-262144}"
ITERS="${ITERS:-200}"
WARMUP="${WARMUP:-20}"
COMPUTE_ELEMS="${COMPUTE_ELEMS:-262144}"
COMPUTE_ITERS="${COMPUTE_ITERS:-20}"
COMPUTE_INNER="${COMPUTE_INNER:-20}"
OUT_DIR="${OUT_DIR:-results}"
RUN_LABEL="${RUN_LABEL:-default}"
MPIEXEC="${MPIEXEC:-mpirun}"
SRUN="${SRUN:-srun}"
MPI_ARGS="${MPI_ARGS:-}"
SRUN_ARGS="${SRUN_ARGS:-}"

mkdir -p "$OUT_DIR"

run_one() {
  local nodes="$1"
  local rpn="$2"
  local total=$((nodes * rpn))
  local stamp
  stamp="$(date +%Y%m%d-%H%M%S)"
  local out="$OUT_DIR/mpi-probe-${RUN_LABEL}-n${nodes}-rpn${rpn}-np${total}-${stamp}.jsonl"

  echo "running label=$RUN_LABEL nodes=$nodes ranks_per_node=$rpn np=$total -> $out" >&2

  local app_args=(
    "$BIN"
    --min-bytes "$MIN_BYTES"
    --max-bytes "$MAX_BYTES"
    --alltoall-max-bytes "$ALLTOALL_MAX_BYTES"
    --iters "$ITERS"
    --warmup "$WARMUP"
    --compute-elems "$COMPUTE_ELEMS"
    --compute-iters "$COMPUTE_ITERS"
    --compute-inner "$COMPUTE_INNER"
  )

  case "$LAUNCHER" in
    slurm)
      # shellcheck disable=SC2206
      local extra_srun_args=($SRUN_ARGS)
      "$SRUN" -N "$nodes" -n "$total" --ntasks-per-node="$rpn" \
        --cpu-bind=cores "${extra_srun_args[@]}" "${app_args[@]}" | tee "$out"
      ;;
    mpirun|openmpi)
      # shellcheck disable=SC2206
      local extra_mpi_args=($MPI_ARGS)
      "$MPIEXEC" -np "$total" --map-by "ppr:${rpn}:node" --bind-to core \
        "${extra_mpi_args[@]}" "${app_args[@]}" | tee "$out"
      ;;
    generic)
      # Use this when MPIEXEC already encodes host allocation and mapping.
      # shellcheck disable=SC2206
      local extra_mpi_args=($MPI_ARGS)
      "$MPIEXEC" -np "$total" "${extra_mpi_args[@]}" "${app_args[@]}" | tee "$out"
      ;;
    *)
      echo "unknown LAUNCHER=$LAUNCHER; use slurm, mpirun, openmpi, or generic" >&2
      exit 2
      ;;
  esac
}

for nodes in $NODES; do
  for rpn in $RANKS_PER_NODE; do
    run_one "$nodes" "$rpn"
  done
done
