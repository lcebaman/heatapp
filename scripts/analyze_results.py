#!/usr/bin/env python3
"""Summarize and compare mpi_probe JSONL result files."""

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict


def load_jsonl(path):
    records = []
    metadata = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{path}:{line_no}: ignoring invalid JSON: {exc}", file=sys.stderr)
                continue
            if "suite" in item and "op" in item:
                item["_file"] = path
                item["_label"] = os.path.splitext(os.path.basename(path))[0]
                records.append(item)
            else:
                item["_file"] = path
                item["_label"] = os.path.splitext(os.path.basename(path))[0]
                metadata.append(item)
    return records, metadata


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize and compare mpi_probe JSONL benchmark results."
    )
    parser.add_argument("files", nargs="+", help="JSONL files produced by mpi_probe")
    parser.add_argument("--suite", help="filter by suite, e.g. p2p, collective, compute, mixed")
    parser.add_argument("--op", help="filter by operation name")
    parser.add_argument("--csv", help="write normalized records to CSV")
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare all files against the first file as baseline",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="maximum table rows per section, default 25",
    )
    return parser.parse_args()


def rate_mib_s(row):
    avg_us = float(row.get("avg_us", 0.0))
    bytes_per_rank = int(row.get("bytes", 0))
    ranks = int(row.get("ranks", 1))
    if avg_us <= 0.0 or bytes_per_rank <= 0:
        return None

    op = row.get("op", "")
    moved = bytes_per_rank
    if op in ("alltoall", "allgather"):
        moved = bytes_per_rank * ranks

    return moved / (1024.0 * 1024.0) / (avg_us / 1.0e6)


def imbalance_pct(row):
    avg_us = float(row.get("avg_us", 0.0))
    if avg_us <= 0.0:
        return 0.0
    return 100.0 * (float(row.get("max_us", avg_us)) - float(row.get("min_us", avg_us))) / avg_us


def normalized_row(row):
    out = {
        "label": row["_label"],
        "file": row["_file"],
        "suite": row.get("suite", ""),
        "op": row.get("op", ""),
        "ranks": int(row.get("ranks", 0)),
        "bytes": int(row.get("bytes", 0)),
        "iters": int(row.get("iters", 0)),
        "ok": bool(row.get("ok", False)),
        "min_us": float(row.get("min_us", 0.0)),
        "avg_us": float(row.get("avg_us", 0.0)),
        "max_us": float(row.get("max_us", 0.0)),
        "imbalance_pct": imbalance_pct(row),
    }
    rate = rate_mib_s(row)
    out["rate_mib_s"] = "" if rate is None else rate
    return out


def filter_records(records, args):
    out = records
    if args.suite:
        out = [r for r in out if r.get("suite") == args.suite]
    if args.op:
        out = [r for r in out if r.get("op") == args.op]
    return out


def fmt_bytes(n):
    n = int(n)
    units = ["B", "KiB", "MiB", "GiB"]
    value = float(n)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{n} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{n} B"


def print_table(headers, rows):
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    print("  ".join("-" * widths[i] for i, _ in enumerate(headers)))
    for row in rows:
        print("  ".join(str(cell).ljust(widths[i]) for i, cell in enumerate(row)))


def summarize(records, metadata, top):
    if not records:
        print("No benchmark records found.")
        return

    failures = [r for r in records if not r.get("ok", False)]
    print(f"records: {len(records)}")
    print(f"correctness failures: {len(failures)}")
    if metadata:
        events = sorted({m.get("event", "metadata") for m in metadata})
        print(f"metadata events: {', '.join(events)}")
    print()

    science_checks = [m for m in metadata if m.get("event") == "science_check"]
    if science_checks:
        rows = []
        for m in science_checks[:top]:
            rows.append(
                [
                    m.get("_label", ""),
                    m.get("op", ""),
                    m.get("steps", ""),
                    m.get("global_points", ""),
                    m.get("nx", ""),
                    m.get("ny", ""),
                    f"{float(m.get('relative_l2_error', math.nan)):.3e}",
                    f"{float(m.get('mass', math.nan)):.3e}",
                    f"{float(m.get('dt', math.nan)):.3e}",
                ]
            )
        print_table(["label", "science_op", "steps", "points", "nx", "ny", "rel_l2", "mass", "dt"], rows)
        print()

    rows = []
    for r in sorted(records, key=lambda x: (x.get("suite", ""), x.get("op", ""), x.get("ranks", 0), x.get("bytes", 0)))[:top]:
        rate = rate_mib_s(r)
        rows.append(
            [
                r["_label"],
                r.get("suite", ""),
                r.get("op", ""),
                r.get("ranks", ""),
                fmt_bytes(r.get("bytes", 0)),
                "yes" if r.get("ok", False) else "NO",
                f"{float(r.get('avg_us', 0.0)):.3f}",
                f"{imbalance_pct(r):.1f}",
                "" if rate is None else f"{rate:.1f}",
            ]
        )
    print_table(
        ["label", "suite", "op", "ranks", "bytes", "ok", "avg_us", "imb_%", "rate_MiB/s"],
        rows,
    )

    if len(records) > top:
        print(f"\nshowing first {top} rows; use --top N to change this")


def compare(records_by_file, top):
    baseline_file = next(iter(records_by_file))
    baseline = {
        (r.get("suite"), r.get("op"), int(r.get("ranks", 0)), int(r.get("bytes", 0))): r
        for r in records_by_file[baseline_file]
    }

    rows = []
    for path, records in list(records_by_file.items())[1:]:
        for r in records:
            key = (r.get("suite"), r.get("op"), int(r.get("ranks", 0)), int(r.get("bytes", 0)))
            b = baseline.get(key)
            if not b:
                continue
            b_avg = float(b.get("avg_us", 0.0))
            avg = float(r.get("avg_us", 0.0))
            if b_avg <= 0.0 or avg <= 0.0:
                continue
            speedup = b_avg / avg
            rows.append(
                [
                    r["_label"],
                    r.get("suite", ""),
                    r.get("op", ""),
                    r.get("ranks", ""),
                    fmt_bytes(r.get("bytes", 0)),
                    f"{b_avg:.3f}",
                    f"{avg:.3f}",
                    f"{speedup:.3f}",
                    "faster" if speedup > 1.02 else "slower" if speedup < 0.98 else "same",
                ]
            )

    rows.sort(key=lambda row: abs(float(row[7]) - 1.0), reverse=True)
    print(f"\ncomparison baseline: {os.path.basename(baseline_file)}")
    if not rows:
        print("No matching records to compare.")
        return
    print_table(
        ["label", "suite", "op", "ranks", "bytes", "base_us", "avg_us", "speedup", "result"],
        rows[:top],
    )
    if len(rows) > top:
        print(f"\nshowing top {top} changes by speedup distance from 1.0")


def write_csv(path, records):
    fields = [
        "label",
        "file",
        "suite",
        "op",
        "ranks",
        "bytes",
        "iters",
        "ok",
        "min_us",
        "avg_us",
        "max_us",
        "imbalance_pct",
        "rate_mib_s",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(normalized_row(r))


def main():
    args = parse_args()
    records_by_file = {}
    all_records = []
    all_metadata = []

    for path in args.files:
        records, metadata = load_jsonl(path)
        records = filter_records(records, args)
        records_by_file[path] = records
        all_records.extend(records)
        all_metadata.extend(metadata)

    summarize(all_records, all_metadata, args.top)

    if args.compare or len(args.files) > 1:
        compare(records_by_file, args.top)

    if args.csv:
        write_csv(args.csv, all_records)
        print(f"\nwrote CSV: {args.csv}")


if __name__ == "__main__":
    main()
