#!/usr/bin/env python3
import subprocess, os, re, statistics, sys
from tqdm import trange
from datetime import datetime
import pandas as pd          # pip install pandas pyarrow
import matplotlib.pyplot as plt
from pathlib import Path
import csv
import json
import numpy as np
from scipy.stats import sem

"""
This script tests the following:
- Overhead-dominated regime (small k): 
  when each writer only pushes a few KB, does the split-comm extra overhead swamp any gains?

- Bandwidth-dominated regime (large k): 
  once you’re writing MBs or 10s of MBs, do you recover enough from fewer-rank collectives to see a clear speedup?

- Crossover point: 
  at roughly what data size do you go from “split_ranks too slow” → “split_ranks wins”?

- Scaling with process count (optional): 
  as you increase from 6 to 12 to 24+ ranks, does the benefit of isolating the writers grow?


Evaluation methods:
- delete output file after each run to avoid caching
- takes median of timings to avoid outliers
"""


# ─── CONFIG ────────────────────────────────────────────────────────────────────
TRIALS = 300                                                                                          # 300
KS     = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]                                 # 1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000
RANKS  = [4, 8, 12]                                                                                   # 4, 8, 12 
CB_CONFIG_LIST = ["*:1","*:2","*:3", "*:4", "*:6", "*:8", "*:12"]                                     # ,"*:2","*:3", "*:4", "*:6", "*:8", "*:12"
SCRIPTS = {
    "all"   : "all_ranks_version.py",
    "split" : "split_ranks_version.py",
}
DEBUG  = False  # set to True to see full output from mpiexec

# Regex to pull out the I/O time
pat_time  = re.compile(r"I/O(?:\(\+split\))? time = ([0-9.]+)")
# Regex for aggregators
pat_cb_nodes = re.compile(r"cb_nodes\s*=\s*(\d+)")
# Regex for cb_config_list
pat_cb_config_list = re.compile(r"cb_config_list\s*=\s*(\S+)")



def run_one(script, ranks, k, cb_config_list, debug=DEBUG):
    """Run one trial: invoke mpiexec with --k, --pairs, --file; parse the timing; delete the output file."""
    # infer number of creator–writer pairs from ranks
    pairs = ranks // 2
    # construct a unique HDF5 filename for this trial
    fname = f"parallel_test_r{ranks}_k{k}.h5"
    # build and run the command
    cmd = [
        "mpiexec",
        "-env", "ROMIO_PRINT_HINTS", "1",
        "-n", str(ranks),
        sys.executable,
        script,
        "--k", str(k),
        "--pairs", str(pairs),
        "--cb_config_list", cb_config_list,
        "--file", fname,
    ]

    try:
        out = subprocess.check_output(
            cmd, text=True,
            stderr=subprocess.STDOUT,
        )
        if debug:
            print(f"\n[DEBUG] Full output from mpiexec:\n{out}\n")
    except subprocess.CalledProcessError as e:
        print("\n❌ MPI command failed!")
        print("Command:", " ".join(cmd))
        print("Exit code:", e.returncode)
        print("Output:\n", e.output)
        raise

    # 2. grab timing ----------------------------------------------------------
    m_time = pat_time.search(out)
    if not m_time:
        raise RuntimeError(f"No timing in output:\n{out}")
    t = float(m_time.group(1))

    # 3. grab aggregator list -------------------------------------------------
    m_aggr = pat_cb_nodes.search(out)
    cb_nodes = int(m_aggr.group(1)) if m_aggr else None

    # 4. grab cb_config_list -----------------------------------------------
    m_config_list = pat_cb_config_list.search(out)
    cb_config_list = m_config_list.group(1) if m_config_list else "unknown"

    # 4. print a quick human-readable note
    if debug:
        print(f"\n [hint] number of cb_nodes (aggregators) used: {cb_nodes}")
        print(f"[hint] cb_config_list: {cb_config_list}")

    # remove the file so the next run starts cold
    if os.path.exists(fname):
        os.remove(fname)
    return t, cb_nodes, cb_config_list

def benchmark(cb_config_list="*:1"):
    results = {}  # results[(script, ranks, k)] = list_of_times
    for script_name, script_file in SCRIPTS.items():
        for ranks in RANKS:
            for k in KS:
                times = []
                cb_nodes_list = []
                cb_config_lists = []
                desc = f"{script_name:5s} R={ranks:2d} k={k:>8d}"
                for _ in trange(TRIALS, desc=desc, unit="run"):
                    t, cb_nodes, cb_config_list = run_one(script_file, ranks, k, cb_config_list)
                    times.append(t)
                    cb_nodes_list.append(cb_nodes)
                    cb_config_lists.append(cb_config_list)

                results[(script_name, ranks, k)] = {
                    "times": times,
                    "cb_nodes_list": cb_nodes_list,
                    "cb_config_lists": cb_config_lists,
                }

    return results

def extract_unique_or_warn_helper(name, ranks, k, lst):
    if all(x == lst[0] for x in lst):
        return lst[0]
    else:
        print(f"[Warning] {name} not unique for ranks={ranks}, k={k}: {lst}")
        return "unknown"


def summarize(results, save=True, outdir="bench_out"):
    print("\n=== SPEEDUP TABLE (median-based) ===")
    header = ["Ranks", "k", "all_med", "split_med", "split/all", "all: cb_nodes | cb_config_list", "split: cb_nodes | cb_config_list"]
    print("{:>5} {:>10} {:>10} {:>10} {:>8} {:>36} {:>36}".format(*header))
    rows = []

    for ranks in RANKS:
        for k in KS:
            all_t  = statistics.median(results[("all", ranks, k)]["times"])
            split_t= statistics.median(results[("split", ranks, k)]["times"])
            speedup= all_t/split_t

            all_cb_nodes        = extract_unique_or_warn_helper("all_cb_nodes", ranks, k, results[("all", ranks, k)]["cb_nodes_list"])
            split_cb_nodes      = extract_unique_or_warn_helper("split_cb_nodes", ranks, k, results[("split", ranks, k)]["cb_nodes_list"])
            all_cb_config_list  = extract_unique_or_warn_helper("all_cb_config_list", ranks, k, results[("all", ranks, k)]["cb_config_lists"])
            split_cb_config_list= extract_unique_or_warn_helper("split_cb_config_list", ranks, k, results[("split", ranks, k)]["cb_config_lists"])


           
            print(f"{ranks:5d} {k:10d} {all_t:10.5f} {split_t:10.5f} {speedup:8.2f}x "
                f"{str(all_cb_nodes):>15}|{str(all_cb_config_list):<20} "
                f"{str(split_cb_nodes):>15}|{str(split_cb_config_list):<20}")

            # Append to CSV
            rows.append([ranks, k, all_t, split_t, speedup, all_cb_nodes, all_cb_config_list, split_cb_nodes, split_cb_config_list])

    # Save to CSV
    if save:
        Path(outdir).mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path_csv = Path(outdir) / f"summarized_table_{timestamp}.csv"
        with open(out_path_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
        print(f"\n💾 Saved summary table → {out_path_csv}")

        out_path_results = Path(outdir) / f"results_{timestamp}.json"
        with open(out_path_results, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)



def materialise(results, outdir="bench_out"):
    """
    Turn the nested results dict into a DataFrame, save to disk,
    create speed-up summary & plots.
    """
    Path(outdir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. flatten: one row = one timing ---------------------------
    rows = []
    for (variant, ranks, k), result_dict in results.items():
        times = result_dict["times"]
        cb_nodes_list = result_dict["cb_nodes_list"]
        cb_config_lists = result_dict["cb_config_lists"]

        for t in times:
            rows.append({
                "variant": variant,
                "ranks": ranks,
                "k": k,
                "time": t,
                "cb_nodes": cb_nodes_list[0],
                "cb_config_list": cb_config_lists[0]
            })
    df = pd.DataFrame(rows)

    # Use the first (ranks, k) combo as reference to extract aggregator info
    ref_ranks = df["ranks"].iloc[0]
    ref_k     = df["k"].iloc[0]

    cb_config_lists = results[("all", ref_ranks, ref_k)]["cb_config_lists"] # cb_config list is constant across ranks and versions so can justr choose this one


    # 2. compute median speed-up table ---------------------------
    med = (df.groupby(["variant", "ranks", "k"])["time"]
             .median()
             .unstack("variant")
             .reset_index())
    med["speedup"] = med["all"] / med["split"]
    # summary_csv = f"{outdir}/summary_{ts}.csv"
    # med.to_csv(summary_csv, index=False)

    # 3. plots ---------------------------------------------------
    plt.figure()
    plt.title(f"Split-writer speed-up vs k (all ranks) - cb_config_list={cb_config_lists[0]}")
    plt.xscale("log")
    plt.xlabel("elements per writer (k)")
    plt.ylabel("speed-up  all/split  ↑ faster")

    # give each ranks value its own line
    for rnk, grp in med.groupby("ranks"):
        all_cb_nodes_list = results[("all", rnk, KS[0])]["cb_nodes_list"]
        split_cb_nodes_list = results[("split", rnk, KS[0])]["cb_nodes_list"]
        label = f"{rnk} ranks (all_ag={all_cb_nodes_list[0]}, split_ag={split_cb_nodes_list[0]})"
        plt.plot(
            grp["k"],                # x-axis
            grp["speedup"],          # y-axis
            marker="o",
            label=label,
        )

    plt.axhline(1.0, ls="--", c="grey", lw=0.8)  # 1× reference line
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(title="MPI ranks")
    plt.tight_layout()

    # save once
    plt.savefig(f"{outdir}/speedup_vs_k_ALL_{ts}.png", dpi=150)
    plt.close()
    
    # print(f"Saved summary  → {summary_csv}")
    print(f"💾 Plots (*.png)  → {outdir}/")


def materialise2(results, outdir="bench_out"):

    Path(outdir).mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Flatten
    rows = []
    for (variant, ranks, k), result_dict in results.items():
        for t in result_dict["times"]:
            rows.append({
                "variant": variant,
                "ranks": ranks,
                "k": k,
                "time": t,
            })
    df = pd.DataFrame(rows)

    # Extract cb_config_list for title
    ref_ranks = df["ranks"].iloc[0]
    ref_k = df["k"].iloc[0]
    cb_config_lists = results[("all", ref_ranks, ref_k)]["cb_config_lists"]

    # Compute median, mean, and std
    grouped = df.groupby(["variant", "ranks", "k"])["time"]
    stats_df = grouped.agg(median="median", mean="mean", std="std", err=sem).reset_index()

    # Pivot for speedup computation
    med = stats_df.pivot(index=["ranks", "k"], columns="variant", values="median").reset_index()
    std = stats_df.pivot(index=["ranks", "k"], columns="variant", values="err").reset_index()

    # Compute speedup and propagated error
    med["speedup"] = med["all"] / med["split"]
    med["speedup_err"] = med["speedup"] * np.sqrt(
        (std["all"] / med["all"])**2 + (std["split"] / med["split"])**2
    )

    # Build label info for consistent legend
    all_cb_nodes_list = {rnk: results[("all", rnk, ref_k)]["cb_nodes_list"][0] for rnk in med["ranks"].unique()}
    split_cb_nodes_list = {rnk: results[("split", rnk, ref_k)]["cb_nodes_list"][0] for rnk in med["ranks"].unique()}

    # Plot
    plt.figure()
    plt.title(f"Split-writer speed-up vs k (all ranks) - cb_config_list={cb_config_lists[0]}")
    plt.xscale("log")
    plt.xlabel("elements per writer (k)")
    plt.ylabel("speed-up  all/split  ↑ faster")

    for rnk, grp in med.groupby("ranks"):
        label = f"{rnk} ranks (all_ag={all_cb_nodes_list[rnk]}, split_ag={split_cb_nodes_list[rnk]})"
        plt.errorbar(
            grp["k"], grp["speedup"], yerr=grp["speedup_err"],
            marker="o", capsize=4, label=label
        )

    plt.axhline(1.0, ls="--", c="grey", lw=0.8)
    plt.grid(True, which="both", ls="--", alpha=0.4)
    plt.legend(title="MPI ranks")
    plt.tight_layout()
    plt.savefig(f"{outdir}/speedup_with_errorbars_{ts}.png", dpi=150)
    plt.close()

    print(f"💾 Saved plot with error bars → {outdir}/speedup_with_errorbars_{ts}.png")



if __name__ == "__main__":
    for cb_config_list in CB_CONFIG_LIST:
        res = benchmark(cb_config_list)
        summarize(res)
        materialise(res)
        materialise2(res)


