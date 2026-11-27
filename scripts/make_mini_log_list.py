#!/usr/bin/env python3
"""Generate a log-name list JSON from a nuPlan split directory.

For nuPlan mini: the upstream training list (nuplan_train.json) only covers the
train part of trainval; mini logs are a mixed subset of trainval, so we build
the authoritative list from the actual .db files present on disk.

Usage:
    python scripts/make_mini_log_list.py data/nuplan/nuplan-v1.1/splits/mini mini_logs.json
"""
import json
import os
import sys


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    split_dir, out_json = sys.argv[1], sys.argv[2]
    logs = sorted(f[:-3] for f in os.listdir(split_dir) if f.endswith(".db"))
    with open(out_json, "w") as f:
        json.dump(logs, f, indent=2)
    print(f"{len(logs)} logs -> {out_json}")
    for name in logs:
        print(" ", name)


if __name__ == "__main__":
    main()
