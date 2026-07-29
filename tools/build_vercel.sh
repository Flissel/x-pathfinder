#!/usr/bin/env bash
# Generate the 3D sample scenes shipped on Vercel.
# Run from repo root: bash tools/build_vercel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/frontend-vercel/sample_scenes"
mkdir -p "$OUT"

run() {
  local kind="$1"; shift
  local hatch="$1"; shift
  # Retuned for visible T gradient in the voxel viewer:
  # P=280W v=800mm/s → E-density ~117 J/mm³ (Ti-64 sweet-spot upper edge),
  # per-frame surface peak reaches ~8500K (Rosenthal singularity artifact;
  # the viewer's inferno colormap clips at boiling ≈3315K for visual clarity).
  # depth_mm=0.15 keeps the volume near the physical LPBF melt-pool depth.
  echo "=== building $kind ==="
  PYTHONPATH="$ROOT" python3 -m laser_sim viz3d \
    --primitive "$kind" --power 280 --speed 800 --hatch "$hatch" --spot 80 \
    --grid-n 24 --n-frames 20 --stride 4 \
    --volume --nz 12 --depth 0.15 \
    --out "$OUT/$kind.json"
}

run zigzag  100
run hilbert 200
run island  100
run voronoi 200

echo
echo "=== built ==="
ls -la "$OUT"

# refresh the static viewer HTML copies from the canonical package files,
# then rewrite the importmap so the Vercel deploy uses the locally-vendored
# three.js under ./vendor/three/ instead of unpkg CDN (avoids CORS/blocklist
# issues and keeps the deploy self-contained).
cp "$ROOT/laser_sim/visualization/three_js_app/volume.html" "$ROOT/frontend-vercel/volume.html"
cp "$ROOT/laser_sim/visualization/three_js_app/index.html"  "$ROOT/frontend-vercel/viewer.html"
cp "$ROOT/laser_sim/visualization/three_js_app/live.html"   "$ROOT/frontend-vercel/live.html"

python3 - "$ROOT/frontend-vercel/volume.html" \
             "$ROOT/frontend-vercel/viewer.html" \
             "$ROOT/frontend-vercel/live.html" <<'PY'
import re, sys
for path in sys.argv[1:]:
    try:
        with open(path) as f: s = f.read()
    except FileNotFoundError:
        continue
    s = re.sub(r'"three":\s*"https://unpkg\.com/[^"]+"',
               '"three": "./vendor/three/three.module.js"', s)
    s = re.sub(r'"three/addons/":\s*"https://unpkg\.com/[^"]+"',
               '"three/addons/": "./vendor/three/jsm/"', s)
    with open(path, "w") as f: f.write(s)
PY

echo
echo "next: cd frontend-vercel && vercel --prod"
