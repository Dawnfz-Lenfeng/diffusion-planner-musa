#!/bin/bash
# Download nuPlan mini + maps + official checkpoint, with byte-size verification,
# retries and resume. Data is NOT bundled with this repo (nuPlan terms apply).
# Usage: bash scripts/download_data.sh [ROOT]   (default: repo dir)
# Produces under ROOT:
#   data/nuplan/maps_zips/{nuplan-maps-v1.0.zip, nuplan-v1.1_mini.zip}
#   checkpoints/{args.json, model.pth}
set -uo pipefail
ROOT=${1:-$(cd "$(dirname "$0")/.." && pwd)}
DATA=$ROOT/data/nuplan
mkdir -p "$DATA/maps_zips" "$ROOT/checkpoints"
CF=https://d1qinkmu0ju04f.cloudfront.net/public/nuplan-v1.1
HF=https://huggingface.co/ZhengYinan2001/Diffusion-Planner/resolve/main

fetch () { # url expect_size(- = skip) out [attempts]
    local url=$1 size=$2 out=$3 attempts=${4:-5} n=1 got
    while [ $n -le $attempts ]; do
        if [ "$size" != "-" ] && [ -f "$out" ] && [ "$(stat -c%s "$out")" = "$size" ]; then
            echo "OK(exists) $out"; return 0
        fi
        wget -c -q -T 30 -t 3 "$url" -O "$out"
        got=$(stat -c%s "$out" 2>/dev/null || echo 0)
        if [ "$size" = "-" ] || [ "$got" = "$size" ]; then
            echo "OK(downloaded) $out ($got bytes)"; return 0
        fi
        echo "RETRY $n/$attempts $out (got $got, want ${size:-any})"; n=$((n+1)); sleep 20
    done
    echo "FAILED $out"; return 1
}

RC=0
fetch "$CF/nuplan-maps-v1.0.zip" 971557640  "$DATA/maps_zips/nuplan-maps-v1.0.zip" || RC=1
fetch "$CF/nuplan-v1.1_mini.zip" 8550100030 "$DATA/maps_zips/nuplan-v1.1_mini.zip" || RC=1
fetch "$HF/args.json" "-"          "$ROOT/checkpoints/args.json" 3 || RC=1
fetch "$HF/model.pth" "48549085"   "$ROOT/checkpoints/model.pth" 3 || RC=1
echo "download rc=$RC"
exit $RC
