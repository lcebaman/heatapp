#include <mpi.h>

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MPI_PROBE_PI 3.141592653589793238462643383279502884

typedef struct {
    size_t min_bytes;
    size_t max_bytes;
    size_t alltoall_max_bytes;
    size_t compute_elems;
    int iters;
    int warmup;
    int compute_iters;
    int compute_inner;
} config_t;

static void die_rank0(int rank, const char *msg) {
    if (rank == 0) {
        fprintf(stderr, "%s\n", msg);
    }
    MPI_Abort(MPI_COMM_WORLD, 2);
}

static void require_mpi_count(size_t n, int rank) {
    if (n > (size_t)INT_MAX) {
        die_rank0(rank, "message size exceeds this MPI build's int count limit");
    }
}

static size_t parse_size(const char *s, int rank) {
    errno = 0;
    char *end = NULL;
    unsigned long long value = strtoull(s, &end, 10);
    if (errno || end == s) {
        die_rank0(rank, "invalid size argument");
    }
    if (*end == 'K' || *end == 'k') {
        value *= 1024ULL;
        end++;
    } else if (*end == 'M' || *end == 'm') {
        value *= 1024ULL * 1024ULL;
        end++;
    } else if (*end == 'G' || *end == 'g') {
        value *= 1024ULL * 1024ULL * 1024ULL;
        end++;
    }
    if (*end != '\0') {
        die_rank0(rank, "invalid size suffix");
    }
    return (size_t)value;
}

static config_t parse_args(int argc, char **argv, int rank) {
    config_t cfg;
    cfg.min_bytes = 8;
    cfg.max_bytes = 1 << 20;
    cfg.alltoall_max_bytes = 1 << 18;
    cfg.compute_elems = 262144;
    cfg.iters = 200;
    cfg.warmup = 20;
    cfg.compute_iters = 20;
    cfg.compute_inner = 20;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min-bytes") == 0 && i + 1 < argc) {
            cfg.min_bytes = parse_size(argv[++i], rank);
        } else if (strcmp(argv[i], "--max-bytes") == 0 && i + 1 < argc) {
            cfg.max_bytes = parse_size(argv[++i], rank);
        } else if (strcmp(argv[i], "--alltoall-max-bytes") == 0 && i + 1 < argc) {
            cfg.alltoall_max_bytes = parse_size(argv[++i], rank);
        } else if (strcmp(argv[i], "--compute-elems") == 0 && i + 1 < argc) {
            cfg.compute_elems = parse_size(argv[++i], rank);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            cfg.iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            cfg.warmup = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compute-iters") == 0 && i + 1 < argc) {
            cfg.compute_iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--compute-inner") == 0 && i + 1 < argc) {
            cfg.compute_inner = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            if (rank == 0) {
                printf("usage: mpi_probe [--min-bytes N] [--max-bytes N] "
                       "[--alltoall-max-bytes N] [--iters N] [--warmup N] "
                       "[--compute-elems N] [--compute-iters N] [--compute-inner N]\n");
            }
            MPI_Finalize();
            exit(0);
        } else {
            die_rank0(rank, "unknown argument; use --help");
        }
    }

    if (cfg.min_bytes == 0 || cfg.max_bytes < cfg.min_bytes || cfg.iters < 1 || cfg.warmup < 0 ||
        cfg.compute_elems < 2 || cfg.compute_iters < 1 || cfg.compute_inner < 1) {
        die_rank0(rank, "invalid benchmark configuration");
    }
    require_mpi_count(cfg.max_bytes, rank);
    require_mpi_count(cfg.alltoall_max_bytes, rank);
    return cfg;
}

static unsigned char pattern_byte(int rank, size_t i, int tag) {
    uint64_t x = (uint64_t)(rank + 1) * 0x9e3779b97f4a7c15ULL;
    x ^= (uint64_t)(i + 17) * 0xbf58476d1ce4e5b9ULL;
    x ^= (uint64_t)(tag + 101) * 0x94d049bb133111ebULL;
    return (unsigned char)((x ^ (x >> 32) ^ (x >> 16)) & 0xffU);
}

static void fill_pattern(unsigned char *buf, size_t n, int rank, int tag) {
    for (size_t i = 0; i < n; i++) {
        buf[i] = pattern_byte(rank, i, tag);
    }
}

static bool check_pattern(const unsigned char *buf, size_t n, int rank, int tag) {
    for (size_t i = 0; i < n; i++) {
        if (buf[i] != pattern_byte(rank, i, tag)) {
            return false;
        }
    }
    return true;
}

static bool global_ok(bool local_ok) {
    int in = local_ok ? 1 : 0;
    int out = 0;
    MPI_Allreduce(&in, &out, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return out == 1;
}

static void emit_result(const char *suite, const char *op, size_t bytes, int ranks, int iters,
                        bool ok, double elapsed) {
    double local_us = elapsed * 1.0e6 / (double)iters;
    double min_us = 0.0;
    double sum_us = 0.0;
    double max_us = 0.0;
    MPI_Reduce(&local_us, &min_us, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_us, &sum_us, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_us, &max_us, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("{\"suite\":\"%s\",\"op\":\"%s\",\"bytes\":%zu,\"ranks\":%d,\"iters\":%d,"
               "\"ok\":%s,\"min_us\":%.3f,\"avg_us\":%.3f,\"max_us\":%.3f}\n",
               suite, op, bytes, ranks, iters, ok ? "true" : "false",
               min_us, sum_us / (double)ranks, max_us);
        fflush(stdout);
    }
}

static void print_run_metadata(int rank, int ranks) {
    if (rank != 0) {
        return;
    }
    const char *names[] = {
        "OMPI_MCA_coll", "OMPI_MCA_coll_ucc_enable", "OMPI_MCA_coll_hcoll_enable",
        "HCOLL_ENABLE_MCAST_ALL", "HCOLL_MAIN_IB", "UCC_TLS", "UCC_CLS",
        "UCX_TLS", "UCX_NET_DEVICES", "MPIR_CVAR_CH4_OFI_ENABLE_HMEM",
        "I_MPI_FABRICS", "I_MPI_COLL_EXTERNAL", "SLURM_JOB_ID", "SLURM_NNODES",
        "SLURM_NTASKS", NULL
    };
    printf("{\"event\":\"environment\"");
    for (int i = 0; names[i] != NULL; i++) {
        const char *v = getenv(names[i]);
        if (v != NULL) {
            printf(",\"%s\":\"%s\"", names[i], v);
        }
    }
    printf("}\n");

    char version[MPI_MAX_LIBRARY_VERSION_STRING];
    int len = 0;
    MPI_Get_library_version(version, &len);
    for (int i = 0; i < len; i++) {
        if (version[i] == '\n' || version[i] == '\r') {
            version[i] = ' ';
        }
    }
    printf("{\"event\":\"mpi_library\",\"version\":\"%.*s\"}\n", len, version);
    printf("{\"event\":\"run\",\"ranks\":%d}\n", ranks);
}

static void print_node_metadata(int rank, int ranks) {
    char name[MPI_MAX_PROCESSOR_NAME + 1];
    int len = 0;
    MPI_Get_processor_name(name, &len);
    name[len] = '\0';

    MPI_Comm local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local_comm);
    int local_rank = 0;
    int local_size = 0;
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_size(local_comm, &local_size);

    char *all_names = NULL;
    int *all_local_ranks = NULL;
    int *all_local_sizes = NULL;
    if (rank == 0) {
        all_names = (char *)calloc((size_t)ranks, MPI_MAX_PROCESSOR_NAME + 1);
        all_local_ranks = (int *)malloc((size_t)ranks * sizeof(int));
        all_local_sizes = (int *)malloc((size_t)ranks * sizeof(int));
        if (!all_names || !all_local_ranks || !all_local_sizes) {
            die_rank0(rank, "allocation failed");
        }
    }

    MPI_Gather(name, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR,
               all_names, MPI_MAX_PROCESSOR_NAME + 1, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_rank, 1, MPI_INT, all_local_ranks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_size, 1, MPI_INT, all_local_sizes, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("{\"event\":\"nodes\",\"leaders\":[");
        int printed = 0;
        for (int r = 0; r < ranks; r++) {
            if (all_local_ranks[r] != 0) {
                continue;
            }
            char *processor = all_names + (size_t)r * (MPI_MAX_PROCESSOR_NAME + 1);
            printf("%s{\"rank\":%d,\"processor\":\"%s\",\"local_ranks\":%d}",
                   printed ? "," : "", r, processor, all_local_sizes[r]);
            printed = 1;
        }
        printf("]}\n");
        fflush(stdout);
        free(all_names);
        free(all_local_ranks);
        free(all_local_sizes);
    }
    MPI_Comm_free(&local_comm);
}

static void bench_pingpong(size_t bytes, const config_t *cfg, int rank, int ranks,
                           unsigned char *sendbuf, unsigned char *recvbuf) {
    if (ranks < 2) {
        return;
    }
    fill_pattern(sendbuf, bytes, rank, 11);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        if (rank == 0) {
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, 100, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 0, 101, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        if (rank == 0) {
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 1, 100, MPI_COMM_WORLD);
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 1, 101, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else if (rank == 1) {
            MPI_Recv(recvbuf, (int)bytes, MPI_BYTE, 0, 100, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(sendbuf, (int)bytes, MPI_BYTE, 0, 101, MPI_COMM_WORLD);
        }
    }
    double elapsed = MPI_Wtime() - t0;
    MPI_Bcast(&elapsed, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    bool ok = true;
    if (rank == 0) {
        ok = check_pattern(recvbuf, bytes, 1, 11);
    } else if (rank == 1) {
        ok = check_pattern(recvbuf, bytes, 0, 11);
    }
    emit_result("p2p", "pingpong_roundtrip", bytes, ranks, cfg->iters, global_ok(ok), elapsed);
}

static void bench_pairwise_sendrecv(size_t bytes, const config_t *cfg, int rank, int ranks,
                                    unsigned char *sendbuf, unsigned char *recvbuf) {
    int peer = rank ^ 1;
    if (peer >= ranks) {
        peer = MPI_PROC_NULL;
    }
    fill_pattern(sendbuf, bytes, rank, 17);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Sendrecv(sendbuf, (int)bytes, MPI_BYTE, peer, 171,
                     recvbuf, (int)bytes, MPI_BYTE, peer, 171,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Sendrecv(sendbuf, (int)bytes, MPI_BYTE, peer, 171,
                     recvbuf, (int)bytes, MPI_BYTE, peer, 171,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = peer == MPI_PROC_NULL || check_pattern(recvbuf, bytes, peer, 17);
    emit_result("p2p", "pairwise_sendrecv", bytes, ranks, cfg->iters, global_ok(ok), elapsed);
}

static void bench_halo(size_t bytes, const config_t *cfg, int rank, int ranks,
                       unsigned char *sendbuf, unsigned char *recvbuf) {
    int left = (rank + ranks - 1) % ranks;
    int right = (rank + 1) % ranks;
    fill_pattern(sendbuf, bytes, rank, 21);

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Request reqs[2];
        MPI_Irecv(recvbuf, (int)bytes, MPI_BYTE, left, 201, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sendbuf, (int)bytes, MPI_BYTE, right, 201, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Request reqs[2];
        MPI_Irecv(recvbuf, (int)bytes, MPI_BYTE, left, 201, MPI_COMM_WORLD, &reqs[0]);
        MPI_Isend(sendbuf, (int)bytes, MPI_BYTE, right, 201, MPI_COMM_WORLD, &reqs[1]);
        MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = check_pattern(recvbuf, bytes, left, 21);
    emit_result("p2p", "nonblocking_ring_halo", bytes, ranks, cfg->iters, global_ok(ok), elapsed);
}

static void bench_bcast(size_t bytes, const config_t *cfg, int rank, int ranks,
                        unsigned char *sendbuf, unsigned char *recvbuf) {
    fill_pattern(sendbuf, bytes, 0, 31);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        if (rank == 0) {
            memcpy(recvbuf, sendbuf, bytes);
        }
        MPI_Bcast(recvbuf, (int)bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        if (rank == 0) {
            memcpy(recvbuf, sendbuf, bytes);
        }
        MPI_Bcast(recvbuf, (int)bytes, MPI_BYTE, 0, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;
    emit_result("collective", "bcast", bytes, ranks, cfg->iters,
                global_ok(check_pattern(recvbuf, bytes, 0, 31)), elapsed);
}

static void bench_allreduce(size_t bytes, const config_t *cfg, int rank, int ranks) {
    int count = (int)(bytes / sizeof(double));
    if (count < 1) {
        count = 1;
    }
    double *in = (double *)malloc((size_t)count * sizeof(double));
    double *out = (double *)malloc((size_t)count * sizeof(double));
    if (!in || !out) {
        die_rank0(rank, "allocation failed");
    }
    for (int i = 0; i < count; i++) {
        in[i] = (double)(rank + 1) + (double)i * 0.001;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Allreduce(in, out, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Allreduce(in, out, count, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = true;
    double rank_sum = (double)ranks * (double)(ranks + 1) / 2.0;
    for (int i = 0; i < count; i++) {
        double expected = rank_sum + (double)ranks * (double)i * 0.001;
        if (fabs(out[i] - expected) > 1.0e-9) {
            ok = false;
            break;
        }
    }
    emit_result("collective", "allreduce_sum_double", (size_t)count * sizeof(double),
                ranks, cfg->iters, global_ok(ok), elapsed);
    free(in);
    free(out);
}

static void bench_reduce(size_t bytes, const config_t *cfg, int rank, int ranks) {
    int count = (int)(bytes / sizeof(int));
    if (count < 1) {
        count = 1;
    }
    int *in = (int *)malloc((size_t)count * sizeof(int));
    int *out = (int *)calloc((size_t)count, sizeof(int));
    if (!in || !out) {
        die_rank0(rank, "allocation failed");
    }
    for (int i = 0; i < count; i++) {
        in[i] = rank + i;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Reduce(in, out, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Reduce(in, out, count, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = true;
    if (rank == 0) {
        int rank_sum = ranks * (ranks - 1) / 2;
        for (int i = 0; i < count; i++) {
            int expected = rank_sum + ranks * i;
            if (out[i] != expected) {
                ok = false;
                break;
            }
        }
    }
    emit_result("collective", "reduce_sum_int", (size_t)count * sizeof(int),
                ranks, cfg->iters, global_ok(ok), elapsed);
    free(in);
    free(out);
}

static void bench_allgather(size_t bytes, const config_t *cfg, int rank, int ranks,
                            unsigned char *sendbuf) {
    fill_pattern(sendbuf, bytes, rank, 41);
    unsigned char *recvbuf = (unsigned char *)malloc(bytes * (size_t)ranks);
    if (!recvbuf) {
        die_rank0(rank, "allocation failed");
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Allgather(sendbuf, (int)bytes, MPI_BYTE, recvbuf, (int)bytes, MPI_BYTE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Allgather(sendbuf, (int)bytes, MPI_BYTE, recvbuf, (int)bytes, MPI_BYTE, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = true;
    for (int r = 0; r < ranks; r++) {
        if (!check_pattern(recvbuf + bytes * (size_t)r, bytes, r, 41)) {
            ok = false;
            break;
        }
    }
    emit_result("collective", "allgather", bytes, ranks, cfg->iters, global_ok(ok), elapsed);
    free(recvbuf);
}

static void bench_alltoall(size_t bytes, const config_t *cfg, int rank, int ranks) {
    if (bytes > cfg->alltoall_max_bytes) {
        return;
    }
    size_t block = bytes;
    unsigned char *sendbuf = (unsigned char *)malloc(block * (size_t)ranks);
    unsigned char *recvbuf = (unsigned char *)malloc(block * (size_t)ranks);
    if (!sendbuf || !recvbuf) {
        die_rank0(rank, "allocation failed");
    }
    for (int r = 0; r < ranks; r++) {
        fill_pattern(sendbuf + block * (size_t)r, block, rank, 500 + r);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Alltoall(sendbuf, (int)block, MPI_BYTE, recvbuf, (int)block, MPI_BYTE, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Alltoall(sendbuf, (int)block, MPI_BYTE, recvbuf, (int)block, MPI_BYTE, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = true;
    for (int r = 0; r < ranks; r++) {
        if (!check_pattern(recvbuf + block * (size_t)r, block, r, 500 + rank)) {
            ok = false;
            break;
        }
    }
    emit_result("collective", "alltoall", bytes, ranks, cfg->iters, global_ok(ok), elapsed);
    free(sendbuf);
    free(recvbuf);
}

static void bench_barrier(const config_t *cfg, int ranks) {
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < cfg->warmup; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double t0 = MPI_Wtime();
    for (int i = 0; i < cfg->iters; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;
    emit_result("collective", "barrier", 0, ranks, cfg->iters, global_ok(true), elapsed);
}

static void init_compute_field(double *a, double *b, size_t n, int rank) {
    for (size_t i = 0; i < n + 2; i++) {
        double x = (double)(i + 1) * 0.000001;
        a[i] = 1.0 + (double)(rank + 1) * 0.01 + x;
        b[i] = 0.5 + (double)(rank + 1) * 0.02 + x * 0.5;
    }
}

static void compute_stencil_steps(double **a_ptr, double **b_ptr, size_t n, int steps) {
    double *a = *a_ptr;
    double *b = *b_ptr;
    for (int step = 0; step < steps; step++) {
        a[0] = a[1];
        a[n + 1] = a[n];
        for (size_t i = 1; i <= n; i++) {
            double left = a[i - 1];
            double center = a[i];
            double right = a[i + 1];
            double lap = left - 2.0 * center + right;
            double grad = right - left;
            double reaction = center * (1.0 - 0.000001 * center);
            b[i] = 0.72 * center + 0.14 * (left + right) + 0.03 * lap +
                   0.01 * grad + 0.10 * reaction;
        }
        double *tmp = a;
        a = b;
        b = tmp;
    }
    *a_ptr = a;
    *b_ptr = b;
}

static bool checksum_field(const double *a, size_t n, double *checksum) {
    double sum = 0.0;
    for (size_t i = 1; i <= n; i++) {
        if (!isfinite(a[i])) {
            return false;
        }
        sum += a[i] * (1.0 + (double)(i % 17) * 0.001);
    }
    *checksum = sum;
    return true;
}

static void bench_compute_only(const config_t *cfg, int rank, int ranks) {
    size_t n = cfg->compute_elems;
    double *a = (double *)malloc((n + 2) * sizeof(double));
    double *b = (double *)malloc((n + 2) * sizeof(double));
    if (!a || !b) {
        die_rank0(rank, "allocation failed");
    }
    init_compute_field(a, b, n, rank);

    MPI_Barrier(MPI_COMM_WORLD);
    compute_stencil_steps(&a, &b, n, cfg->warmup > 0 ? 1 : 0);
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int iter = 0; iter < cfg->compute_iters; iter++) {
        compute_stencil_steps(&a, &b, n, cfg->compute_inner);
    }
    double elapsed = MPI_Wtime() - t0;

    double local_checksum = 0.0;
    bool ok = checksum_field(a, n, &local_checksum);
    double global_checksum = 0.0;
    MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    ok = ok && isfinite(global_checksum) && global_checksum > 0.0;
    emit_result("compute", "local_stencil", n * sizeof(double), ranks,
                cfg->compute_iters, global_ok(ok), elapsed);
    free(a);
    free(b);
}

static void bench_compute_halo_allreduce(const config_t *cfg, int rank, int ranks) {
    size_t n = cfg->compute_elems;
    double *a = (double *)malloc((n + 2) * sizeof(double));
    double *b = (double *)malloc((n + 2) * sizeof(double));
    if (!a || !b) {
        die_rank0(rank, "allocation failed");
    }
    init_compute_field(a, b, n, rank);

    int left = (rank + ranks - 1) % ranks;
    int right = (rank + 1) % ranks;
    double global_checksum = 0.0;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int iter = 0; iter < cfg->compute_iters; iter++) {
        MPI_Sendrecv(&a[1], 1, MPI_DOUBLE, left, 701,
                     &a[n + 1], 1, MPI_DOUBLE, right, 701,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&a[n], 1, MPI_DOUBLE, right, 702,
                     &a[0], 1, MPI_DOUBLE, left, 702,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        compute_stencil_steps(&a, &b, n, cfg->compute_inner);
        double local_checksum = 0.0;
        bool finite = checksum_field(a, n, &local_checksum);
        if (!finite) {
            local_checksum = -1.0;
        }
        MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    }
    double elapsed = MPI_Wtime() - t0;

    bool ok = isfinite(global_checksum) && global_checksum > 0.0;
    emit_result("mixed", "stencil_halo_allreduce", n * sizeof(double), ranks,
                cfg->compute_iters, global_ok(ok), elapsed);
    free(a);
    free(b);
}

static size_t idx2(size_t y, size_t x, size_t nx) {
    return y * nx + x;
}

static void bench_science_heat2d(const config_t *cfg, int rank, int ranks) {
    size_t target_cells = cfg->compute_elems;
    size_t nx = (size_t)sqrt((double)target_cells);
    if (nx < 4) {
        nx = 4;
    }
    size_t local_ny = target_cells / nx;
    if (local_ny < 4) {
        local_ny = 4;
    }
    require_mpi_count(nx, rank);
    size_t local_cells = nx * local_ny;
    size_t global_ny = local_ny * (size_t)ranks;
    size_t global_cells = nx * global_ny;
    int steps = cfg->compute_iters * cfg->compute_inner;
    double alpha = 0.1;
    double dx = 1.0 / (double)nx;
    double dy = 1.0 / (double)global_ny;
    double min_h2 = dx < dy ? dx * dx : dy * dy;
    double dt = 0.2 * min_h2 / alpha;
    double coeff_x = alpha * dt / (dx * dx);
    double coeff_y = alpha * dt / (dy * dy);
    double wave = 2.0 * MPI_PROBE_PI;

    double *u = (double *)malloc((local_ny + 2) * nx * sizeof(double));
    double *v = (double *)malloc((local_ny + 2) * nx * sizeof(double));
    if (!u || !v) {
        die_rank0(rank, "allocation failed");
    }

    for (size_t y = 0; y < local_ny + 2; y++) {
        for (size_t x = 0; x < nx; x++) {
            u[idx2(y, x, nx)] = 0.0;
            v[idx2(y, x, nx)] = 0.0;
        }
    }

    for (size_t y = 1; y <= local_ny; y++) {
        size_t gy = (size_t)rank * local_ny + (y - 1);
        double yy = ((double)gy + 0.5) * dy;
        for (size_t x = 0; x < nx; x++) {
            double xx = ((double)x + 0.5) * dx;
            u[idx2(y, x, nx)] = sin(wave * xx) * sin(wave * yy);
        }
    }

    int top = (rank + ranks - 1) % ranks;
    int bottom = (rank + 1) % ranks;

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int step = 0; step < steps; step++) {
        MPI_Sendrecv(&u[idx2(1, 0, nx)], (int)nx, MPI_DOUBLE, top, 801,
                     &u[idx2(local_ny + 1, 0, nx)], (int)nx, MPI_DOUBLE, bottom, 801,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Sendrecv(&u[idx2(local_ny, 0, nx)], (int)nx, MPI_DOUBLE, bottom, 802,
                     &u[idx2(0, 0, nx)], (int)nx, MPI_DOUBLE, top, 802,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (size_t y = 1; y <= local_ny; y++) {
            for (size_t x = 0; x < nx; x++) {
                size_t xm = x == 0 ? nx - 1 : x - 1;
                size_t xp = x + 1 == nx ? 0 : x + 1;
                double center = u[idx2(y, x, nx)];
                double lap_x = u[idx2(y, xm, nx)] - 2.0 * center + u[idx2(y, xp, nx)];
                double lap_y = u[idx2(y - 1, x, nx)] - 2.0 * center + u[idx2(y + 1, x, nx)];
                v[idx2(y, x, nx)] = center + coeff_x * lap_x + coeff_y * lap_y;
            }
        }
        double *tmp = u;
        u = v;
        v = tmp;
    }
    double elapsed = MPI_Wtime() - t0;

    double final_t = dt * (double)steps;
    double decay = exp(-alpha * (wave * wave + wave * wave) * final_t);
    double local_err2 = 0.0;
    double local_exact2 = 0.0;
    double local_mass = 0.0;
    bool finite = true;

    for (size_t y = 1; y <= local_ny; y++) {
        size_t gy = (size_t)rank * local_ny + (y - 1);
        double yy = ((double)gy + 0.5) * dy;
        for (size_t x = 0; x < nx; x++) {
            double xx = ((double)x + 0.5) * dx;
            double exact = sin(wave * xx) * sin(wave * yy) * decay;
            double value = u[idx2(y, x, nx)];
            double diff = value - exact;
            if (!isfinite(value)) {
                finite = false;
            }
            local_err2 += diff * diff;
            local_exact2 += exact * exact;
            local_mass += value;
        }
    }

    double global_err2 = 0.0;
    double global_exact2 = 0.0;
    double global_mass = 0.0;
    MPI_Allreduce(&local_err2, &global_err2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_exact2, &global_exact2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_mass, &global_mass, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double rel_l2 = sqrt(global_err2 / global_exact2);
    bool ok = finite && isfinite(rel_l2) && rel_l2 < 5.0e-3 &&
              fabs(global_mass) < 1.0e-10 * (double)global_cells;
    emit_result("science", "heat2d_periodic", local_cells * sizeof(double),
                ranks, steps, global_ok(ok), elapsed);

    if (rank == 0) {
        printf("{\"event\":\"science_check\",\"op\":\"heat2d_periodic\","
               "\"relative_l2_error\":%.12e,\"mass\":%.12e,\"steps\":%d,"
               "\"global_points\":%zu,\"nx\":%zu,\"ny\":%zu,\"local_ny\":%zu,"
               "\"dt\":%.12e}\n",
               rel_l2, global_mass, steps, global_cells, nx, global_ny, local_ny, dt);
        fflush(stdout);
    }

    free(u);
    free(v);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int ranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    config_t cfg = parse_args(argc, argv, rank);
    print_run_metadata(rank, ranks);
    print_node_metadata(rank, ranks);

    unsigned char *sendbuf = (unsigned char *)malloc(cfg.max_bytes);
    unsigned char *recvbuf = (unsigned char *)malloc(cfg.max_bytes);
    if (!sendbuf || !recvbuf) {
        die_rank0(rank, "allocation failed");
    }

    bench_compute_only(&cfg, rank, ranks);
    bench_compute_halo_allreduce(&cfg, rank, ranks);
    bench_science_heat2d(&cfg, rank, ranks);
    bench_barrier(&cfg, ranks);
    for (size_t bytes = cfg.min_bytes; bytes <= cfg.max_bytes; bytes *= 2) {
        bench_pingpong(bytes, &cfg, rank, ranks, sendbuf, recvbuf);
        bench_pairwise_sendrecv(bytes, &cfg, rank, ranks, sendbuf, recvbuf);
        bench_halo(bytes, &cfg, rank, ranks, sendbuf, recvbuf);
        bench_bcast(bytes, &cfg, rank, ranks, sendbuf, recvbuf);
        bench_reduce(bytes, &cfg, rank, ranks);
        bench_allreduce(bytes, &cfg, rank, ranks);
        bench_allgather(bytes, &cfg, rank, ranks, sendbuf);
        bench_alltoall(bytes, &cfg, rank, ranks);
        if (bytes > cfg.max_bytes / 2) {
            break;
        }
    }

    free(sendbuf);
    free(recvbuf);
    MPI_Finalize();
    return 0;
}
