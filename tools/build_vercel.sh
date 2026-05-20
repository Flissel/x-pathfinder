#!/usr/bin/env bash
# Generate the 3D sample scenes shipped on Vercel.
# Run from repo root: bash tools/build_vercel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/frontend-vercel/public/sample_scenes"
mkdir -p "$OUT"

run() {
  local kind="$1"; shift
  local hatch="$1"; shift
  echo "=== building $kind ==="
  PYTHONPATH="$ROOT" python3 -m laser_sim viz3d \
    --primitive "$kind" --power 220 --speed 1000 --hatch "$hatch" --spot 80 \
    --grid-n 24 --n-frames 16 --stride 6 \
    --volume --nz 12 --depth 0.4 \
    --out "$OUT/$kind.json"
}

run zigzag  100
run hilbert 200
run island  100
run voronoi 200

echo
echo "=== built ==="
ls -la "$OUT"

# refresh the static viewer HTML copies from the canonical package files
cp "$ROOT/laser_sim/visualization/three_js_app/volume.html" "$ROOT/frontend-vercel/volume.html"
cp "$ROOT/laser_sim/visualization/three_js_app/index.html"  "$ROOT/frontend-vercel/viewer.html"
cp "$ROOT/laser_sim/visualization/three_js_app/live.html"   "$ROOT/frontend-vercel/live.html"

echo
echo "next: cd frontend-vercel && vercel --prod"
