#!/usr/bin/env bash
#SBATCH --job-name=mpi-probe
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00
#SBATCH --output=results/slurm-%j.out

set -euo pipefail

mkdir -p results

# Load your MPI module before running, for example:
# module purge
# module load openmpi

make

LAUNCHER=slurm \
NODES="${NODES:-1 2}" \
RANKS_PER_NODE="${RANKS_PER_NODE:-1 2 4 8}" \
RUN_LABEL="${RUN_LABEL:-slurm-${SLURM_JOB_ID}}" \
MIN_BYTES="${MIN_BYTES:-8}" \
MAX_BYTES="${MAX_BYTES:-1048576}" \
ALLTOALL_MAX_BYTES="${ALLTOALL_MAX_BYTES:-262144}" \
ITERS="${ITERS:-200}" \
COMPUTE_ELEMS="${COMPUTE_ELEMS:-262144}" \
COMPUTE_ITERS="${COMPUTE_ITERS:-20}" \
COMPUTE_INNER="${COMPUTE_INNER:-20}" \
scripts/run_node_matrix.sh
