# MPI Correctness and Performance Harness

This project is a small scientific-style MPI application for comparing MPI
implementations and collective offload stacks such as HCOLL and UCC. It runs
repeatable correctness checks and timing measurements for point-to-point and
collective operations.

The benchmark is intentionally application-shaped rather than a raw microbench:
each rank owns a deterministic vector field, exchanges halo-like data, and
checks global reductions against known analytic results. This makes it useful
for catching both correctness regressions and performance changes while swapping
MPI libraries.

## Features

- Point-to-point tests:
  - blocking ping-pong latency and bandwidth
  - nonblocking halo exchange with validation
  - all-rank ring exchange
- Compute tests:
  - local finite-difference-style stencil kernel
  - mixed stencil, halo exchange, and `MPI_Allreduce` iteration
- Science test:
  - distributed periodic 2D heat equation
  - halo exchange at subdomain boundaries
  - analytic sine-wave decay check using relative L2 error
  - global mass conservation check
- Collective tests:
  - `MPI_Bcast`
  - `MPI_Reduce`
  - `MPI_Allreduce`
  - `MPI_Allgather`
  - `MPI_Alltoall`
  - `MPI_Barrier`
- Correctness checks with deterministic payloads.
- JSON-lines output for easy comparison across MPI implementations.
- Node metadata using `MPI_COMM_TYPE_SHARED`, so multi-node runs show one
  leader record per node and the number of local ranks on that node.
- Environment capture for MPI, HCOLL, UCC, UCX, OMPI, MPICH, Intel MPI, and
  Slurm-related variables.

## Build

Use the `mpicc` from the MPI implementation you want to test:

```bash
mpicc -O3 -std=c11 -Wall -Wextra -o mpi_probe src/mpi_probe.c -lm
```

Or with the included makefile:

```bash
make
```

## Run

```bash
mpirun -np 4 ./mpi_probe --min-bytes 8 --max-bytes 1048576 --iters 200
```

To increase the expensive compute portion:

```bash
mpirun -np 16 ./mpi_probe --compute-elems 1048576 --compute-iters 50 --compute-inner 40
```

Example JSON-line result:

```json
{"suite":"collective","op":"allreduce","bytes":8192,"ranks":4,"iters":200,"ok":true,"min_us":12.4,"avg_us":13.1,"max_us":14.8}
```

## Testing HCOLL and UCC

The harness does not call HCOLL or UCC directly. Instead, it drives standard MPI
operations and records the environment, so you can compare the MPI stack with
different collective components enabled.

Open MPI with UCC, example:

```bash
mpirun -np 8 --mca coll_ucc_enable 1 --mca coll ^hcoll ./mpi_probe
```

Open MPI or HPC-X with HCOLL, example:

```bash
mpirun -np 8 -x HCOLL_ENABLE_MCAST_ALL=1 --mca coll_hcoll_enable 1 ./mpi_probe
```

UCX transport settings, example:

```bash
mpirun -np 8 -x UCX_TLS=rc,sm,self ./mpi_probe
```

For production comparisons, pin CPU placement and host allocation outside the
program so each MPI implementation gets the same resources.

## Matrix Runner

```bash
BIN=./mpi_probe RANKS="2 4 8 16" scripts/run_matrix.sh
```

For multi-node experiments, sweep both node count and ranks per node:

```bash
LAUNCHER=slurm \
NODES="1 2 4" \
RANKS_PER_NODE="1 2 4 8" \
RUN_LABEL=openmpi-default \
scripts/run_node_matrix.sh
```

Open MPI without Slurm:

```bash
LAUNCHER=mpirun \
NODES="1 2" \
RANKS_PER_NODE="4 8" \
RUN_LABEL=openmpi-ucc \
MPI_ARGS="--mca coll_ucc_enable 1 --mca coll ^hcoll" \
scripts/run_node_matrix.sh
```

## Reading Results

Each benchmark line is one JSON object. Treat `ok` as the first signal: if it is
`false`, the run found a correctness problem and the timing should not be used
for performance conclusions.

Timing fields are microseconds per timed iteration:

- `avg_us`: average per-rank time; lower is better.
- `min_us`: fastest rank time.
- `max_us`: slowest rank time.
- Large `max_us - min_us` gaps usually mean imbalance, noisy placement, network
  contention, or one rank taking a different path.

For communication tests, compare the same `suite`, `op`, `ranks`, and `bytes`
across MPI implementations. For compute tests, `bytes` is the per-rank working
set size, not a message payload. The `mixed` suite is often the most useful
application-style signal because it combines compute, neighbor exchange, and a
global reduction.

Quick summary:

```bash
python3 scripts/analyze_results.py results/mpi-probe-np16-default.jsonl
```

Compare a UCC or HCOLL run against a baseline. The first file is the baseline;
speedup above `1.0` means the later file was faster:

```bash
python3 scripts/analyze_results.py \
  results/mpi-probe-np16-default.jsonl \
  results/mpi-probe-np16-ucc.jsonl \
  --compare
```

Filter to one operation and export CSV:

```bash
python3 scripts/analyze_results.py results/*.jsonl \
  --suite collective --op allreduce --csv allreduce.csv
```

Plot figures as PNG files:

```bash
python3 scripts/plot_results.py results/*.jsonl --out-dir figures
```

Plot only one operation:

```bash
python3 scripts/plot_results.py results/*.jsonl \
  --suite collective --op allreduce --kind latency
```

Plot speedup against a baseline run:

```bash
python3 scripts/plot_results.py \
  results/mpi-probe-openmpi-default-*.jsonl \
  results/mpi-probe-openmpi-ucc-*.jsonl \
  --kind speedup \
  --baseline-label mpi-probe-openmpi-default-n2-rpn8-np16
```

`plot_results.py` requires `matplotlib`:

```bash
python3 -m pip install matplotlib
```

The analyzer adds:

- `imb_%`: `(max_us - min_us) / avg_us * 100`, a quick imbalance indicator.
- `rate_MiB/s`: an effective payload rate for communication-oriented rows.
- `speedup`: baseline `avg_us` divided by candidate `avg_us`.

## Multi-Node Scalability

For a few nodes, keep launch policy outside the application so each MPI
implementation gets the same allocation, mapping, and binding. Examples:

```bash
mpirun -np 16 --map-by ppr:8:node --bind-to core ./mpi_probe
```

With Slurm:

```bash
srun -N 2 -n 16 --ntasks-per-node=8 --cpu-bind=cores ./mpi_probe
```

The important placement variables are:

- nodes: number of physical nodes in the allocation.
- ranks per node: MPI processes placed on each node.
- total ranks: `nodes * ranks_per_node`.
- binding: pin ranks to cores consistently, for example `--bind-to core` or
  `--cpu-bind=cores`.

Use the node matrix runner to create comparable result files:

```bash
LAUNCHER=slurm NODES="1 2 4" RANKS_PER_NODE="1 4 8" scripts/run_node_matrix.sh
```

Output filenames include `n<NODES>`, `rpn<RANKS_PER_NODE>`, and `np<TOTAL>`,
which makes later comparisons less error-prone.

Submit the included Slurm template after editing the `#SBATCH` lines for your
cluster:

```bash
sbatch scripts/slurm_node_matrix.sbatch
```

Make sure the Slurm allocation is large enough for the largest requested matrix
entry. For example, `NODES="1 2 4"` with `RANKS_PER_NODE="1 4 8"` needs an
allocation of at least 4 nodes and 8 tasks per node.

The pairwise point-to-point test uses all ranks in paired `MPI_Sendrecv`
exchanges, and the halo test exercises a full rank ring. `MPI_Alltoall` is
capped by `--alltoall-max-bytes` because its memory and network volume grow as
`ranks * bytes` per rank. Raise the cap when the node count and memory budget
can support it.

The mixed compute test runs an iterative stencil, exchanges rank-boundary values
with neighbors, and performs a checksum `MPI_Allreduce` every outer iteration.
Use `--compute-elems` to set per-rank working-set size, `--compute-iters` for
outer iterations, and `--compute-inner` for stencil steps between MPI calls.

The `science` suite solves the periodic 2D heat equation:

```text
du/dt = alpha * (d2u/dx2 + d2u/dy2)
```

The initial condition is `sin(2*pi*x) * sin(2*pi*y)`, whose exact solution
decays as `exp(-alpha * ((2*pi)^2 + (2*pi)^2) * t)`. Each rank owns a horizontal
slab of rows, uses periodic wrap in the x direction, exchanges top/bottom halo
rows with neighbor ranks every time step, and checks the final field against the
analytic answer. The benchmark emits both a timing row and a metadata row like:

```json
{"event":"science_check","op":"heat2d_periodic","relative_l2_error":1.2e-6,"mass":2.1e-14,"steps":400,"global_points":4194304,"nx":2048,"ny":2048,"local_ny":256,"dt":1.4e-7}
```

This gives you a real numerical correctness signal in addition to MPI API
correctness.

## Recommended Experiment Matrix

Run each MPI stack with the same rank counts and message sizes:

- rank counts: `2, 4, 8, 16, 32, 64`
- message sizes: `8 B` through `64 MiB`
- iterations: `1000` for latency, `100` to `200` for large messages
- placement: same nodes, same ranks per node, same CPU binding
- transports: record UCX, OFI, shared-memory, and GPU settings separately

Good comparisons:

- Open MPI default collectives vs Open MPI UCC.
- HPC-X/Open MPI default vs HCOLL vs UCC.
- MPICH or Intel MPI baseline vs Open MPI variants.
- single-node shared memory vs multi-node fabric.

## Output Fields

- `suite`: `p2p`, `collective`, `compute`, `mixed`, or `science`
- `op`: operation name
- `bytes`: payload or compute-field size per rank, where applicable
- `ranks`: `MPI_COMM_WORLD` size
- `iters`: timed iterations
- `ok`: correctness status across all ranks
- `min_us`, `avg_us`, `max_us`: per-iteration time in microseconds, reduced
  across ranks

## Notes

- Timings use `MPI_Wtime`.
- Each timed loop includes a barrier before measurement to improve comparability.
- Correctness checks are run for every tested message size.
- The code avoids MPI implementation-specific APIs so the same source can be
  compiled by any conforming MPI compiler wrapper.
