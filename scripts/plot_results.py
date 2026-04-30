#!/usr/bin/env python3
"""Plot mpi_probe JSONL result files."""

import argparse
import json
import os
import sys
from collections import defaultdict


def load_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        print(
            "matplotlib is required. Install it with: python3 -m pip install matplotlib",
            file=sys.stderr,
        )
        raise


def load_jsonl(path):
    records = []
    metadata = []
    label = os.path.splitext(os.path.basename(path))[0]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            item["_file"] = path
            item["_label"] = label
            if "suite" in item and "op" in item:
                records.append(item)
            else:
                metadata.append(item)
    return records, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Create PNG plots from mpi_probe JSONL files.")
    parser.add_argument("files", nargs="+", help="JSONL files produced by mpi_probe")
    parser.add_argument("--out-dir", default="figures", help="output directory, default figures")
    parser.add_argument("--suite", help="filter suite, e.g. collective, p2p, science")
    parser.add_argument("--op", help="filter operation")
    parser.add_argument(
        "--kind",
        choices=["latency", "speedup", "imbalance", "science", "all"],
        default="all",
        help="plot type, default all",
    )
    parser.add_argument("--baseline", help="baseline JSONL file for speedup plots")
    parser.add_argument("--baseline-label", help="baseline file label for speedup plots")
    return parser.parse_args()


def filter_records(records, suite=None, op=None):
    out = records
    if suite:
        out = [r for r in out if r.get("suite") == suite]
    if op:
        out = [r for r in out if r.get("op") == op]
    return out


def safe_name(text):
    keep = []
    for ch in text:
        keep.append(ch if ch.isalnum() or ch in ("-", "_") else "_")
    return "".join(keep).strip("_")


def group_by_op(records):
    groups = defaultdict(list)
    for r in records:
        groups[(r.get("suite", ""), r.get("op", ""))].append(r)
    return groups


def imbalance_pct(row):
    avg = float(row.get("avg_us", 0.0))
    if avg <= 0.0:
        return 0.0
    return 100.0 * (float(row.get("max_us", avg)) - float(row.get("min_us", avg))) / avg


def plot_latency(plt, records, out_dir):
    for (suite, op), rows in group_by_op(records).items():
        rows = sorted(rows, key=lambda r: (int(r.get("ranks", 0)), int(r.get("bytes", 0))))
        by_label_rank = defaultdict(list)
        for r in rows:
            by_label_rank[(r["_label"], int(r.get("ranks", 0)))].append(r)

        fig, ax = plt.subplots(figsize=(9, 5))
        for (label, ranks), values in sorted(by_label_rank.items()):
            values = sorted(values, key=lambda r: int(r.get("bytes", 0)))
            xs = [max(1, int(r.get("bytes", 0))) for r in values]
            ys = [float(r.get("avg_us", 0.0)) for r in values]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{label} np={ranks}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("bytes per rank")
        ax.set_ylabel("avg_us per iteration")
        ax.set_title(f"{suite}/{op}: latency")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = os.path.join(out_dir, f"latency-{safe_name(suite)}-{safe_name(op)}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(out)


def plot_imbalance(plt, records, out_dir):
    for (suite, op), rows in group_by_op(records).items():
        rows = sorted(rows, key=lambda r: (r["_label"], int(r.get("ranks", 0)), int(r.get("bytes", 0))))
        fig, ax = plt.subplots(figsize=(9, 5))
        by_label = defaultdict(list)
        for r in rows:
            by_label[r["_label"]].append(r)
        for label, values in sorted(by_label.items()):
            xs = [max(1, int(r.get("bytes", 0))) for r in values]
            ys = [imbalance_pct(r) for r in values]
            ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
        ax.set_xscale("log")
        ax.set_xlabel("bytes per rank")
        ax.set_ylabel("imbalance %")
        ax.set_title(f"{suite}/{op}: rank imbalance")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = os.path.join(out_dir, f"imbalance-{safe_name(suite)}-{safe_name(op)}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(out)


def plot_speedup(plt, records, baseline_path, baseline_label_arg, out_dir):
    if baseline_label_arg:
        baseline_label = baseline_label_arg
    elif not baseline_path:
        baseline_label = records[0]["_label"] if records else ""
    else:
        baseline_label = os.path.splitext(os.path.basename(baseline_path))[0]

    baseline = {
        (r.get("suite"), r.get("op"), int(r.get("ranks", 0)), int(r.get("bytes", 0))): r
        for r in records
        if r["_label"] == baseline_label
    }
    if not baseline:
        print("No baseline records found for speedup plot.", file=sys.stderr)
        return

    for (suite, op), rows in group_by_op(records).items():
        fig, ax = plt.subplots(figsize=(9, 5))
        by_label = defaultdict(list)
        for r in rows:
            if r["_label"] == baseline_label:
                continue
            key = (r.get("suite"), r.get("op"), int(r.get("ranks", 0)), int(r.get("bytes", 0)))
            b = baseline.get(key)
            if not b:
                continue
            b_avg = float(b.get("avg_us", 0.0))
            avg = float(r.get("avg_us", 0.0))
            if b_avg <= 0.0 or avg <= 0.0:
                continue
            by_label[r["_label"]].append((max(1, int(r.get("bytes", 0))), b_avg / avg))

        if not by_label:
            plt.close(fig)
            continue

        for label, values in sorted(by_label.items()):
            values.sort()
            ax.plot([v[0] for v in values], [v[1] for v in values], marker="o", label=label)
        ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--")
        ax.set_xscale("log")
        ax.set_xlabel("bytes per rank")
        ax.set_ylabel("speedup vs baseline")
        ax.set_title(f"{suite}/{op}: speedup vs {baseline_label}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize="small")
        fig.tight_layout()
        out = os.path.join(out_dir, f"speedup-{safe_name(suite)}-{safe_name(op)}.png")
        fig.savefig(out, dpi=160)
        plt.close(fig)
        print(out)


def plot_science(plt, metadata, out_dir):
    rows = [m for m in metadata if m.get("event") == "science_check"]
    if not rows:
        return
    rows.sort(key=lambda m: (m.get("_label", ""), int(m.get("global_points", 0))))

    fig, ax = plt.subplots(figsize=(9, 5))
    by_label = defaultdict(list)
    for m in rows:
        by_label[m["_label"]].append(m)
    for label, values in sorted(by_label.items()):
        xs = [int(m.get("global_points", 0)) for m in values]
        ys = [float(m.get("relative_l2_error", 0.0)) for m in values]
        ax.plot(xs, ys, marker="o", linewidth=1.5, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("global grid points")
    ax.set_ylabel("relative L2 error")
    ax.set_title("science check: 2D heat equation error")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize="small")
    fig.tight_layout()
    out = os.path.join(out_dir, "science-heat2d-error.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print(out)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records = []
    metadata = []
    for path in args.files:
        r, m = load_jsonl(path)
        records.extend(r)
        metadata.extend(m)

    records = filter_records(records, args.suite, args.op)
    if not records and args.kind != "science":
        print("No matching benchmark records.", file=sys.stderr)
        return 1

    plt = load_pyplot()
    if args.kind in ("latency", "all"):
        plot_latency(plt, records, args.out_dir)
    if args.kind in ("imbalance", "all"):
        plot_imbalance(plt, records, args.out_dir)
    if args.kind in ("speedup", "all") and len({r["_label"] for r in records}) > 1:
        plot_speedup(plt, records, args.baseline, args.baseline_label, args.out_dir)
    if args.kind in ("science", "all"):
        plot_science(plt, metadata, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
