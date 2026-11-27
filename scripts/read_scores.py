#!/usr/bin/env python3
"""Read aggregated closed-loop scores from a metric run dir.
Usage: python read_scores.py <metric_run_dir> [runner_report.parquet]
"""
import glob
import sys

import pandas as pd

metric_dir = sys.argv[1]
runner = sys.argv[2] if len(sys.argv) > 2 else None

agg = glob.glob(f"{metric_dir}/aggregator_metric/*.parquet")
print(f"aggregator files: {len(agg)}")
for f in agg:
    df = pd.read_parquet(f)
    show = [c for c in ["scenario_type", "num_scenarios", "score", "aggregator_type"] if c in df.columns]
    print(df[show].to_string(index=False))
    extra = [c for c in df.columns if "compliance" in c or "collision" in c or "comfortable" in c or "progress" in c or "tti" in c.lower()][:8]
    if extra and "final_score" in df["scenario_type"].values:
        fin = df[df["scenario_type"] == "final_score"]
        print("\nfinal_score row detail:")
        print(fin[["scenario_type"] + extra].to_string(index=False))

files = glob.glob(f"{metric_dir}/metrics/*.parquet")
tokens = set()
for f in files:
    try:
        df = pd.read_parquet(f)
        if "scenario" in df.columns:
            tokens.update(df["scenario"].unique())
        elif "log_name" in df.columns:
            tokens.update(df["log_name"].unique())
    except Exception as e:
        print(f"skip {f}: {e}")
print(f"\nmetric files: {len(files)}, distinct scenarios covered: {len(tokens)}")

if runner:
    try:
        df = pd.read_parquet(runner)
        print("\nrunner_report status:")
        print(df["status"].value_counts().to_string())
    except Exception as e:
        print(f"runner_report: {e}")
