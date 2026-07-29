# laser_sim · Vercel frontend

Static deployment of the Three.js volumetric viewer. No backend, no functions, no DB.

## Layout

```
frontend-vercel/
  index.html               landing page with viewer links
  volume.html              3D voxel viewer (true volumetric, schema v2)
  viewer.html              2.5D heightfield viewer (schema v1 fallback)
  live.html                SSE evolution dashboard (needs local backend)
  vercel.json              static-only config, CORS on /sample_scenes/
  public/
    sample_scenes/
      zigzag.json          pre-computed 3D scenes
      hilbert.json         24×24×12 voxels, 16 frames each, Ti-64
      island.json
      voronoi.json
```

## Deploy

```bash
cd frontend-vercel
vercel --prod
```

Vercel detects static automatically (no `builds`/`functions`). The sample scenes
are served with 1h `Cache-Control: immutable` + CORS so external clients can
load them too.

## Regenerate sample scenes

```bash
bash ../tools/build_vercel.sh
```

Runs `python -m laser_sim viz3d --volume` for four representative scan patterns
and writes them into `sample_scenes/`. Re-run whenever the physics or
the scene-export schema changes.

## URL parameters (volume.html)

| Param | Default | Effect |
|---|---|---|
| `?scene=<url>` | `/scene.json` | Static mode — fetch scene JSON v2 |
| `?live=<sse-url>` | none | Live mode — subscribe to SSE `volume_frame` stream |
| `?t_iso=<0..1>` | `0` | Hide voxels below this fraction of (vmax-vmin) |
| `?point=<px>` | `14` | Base point sprite size |

## Static mode (typical Vercel usage)

```
https://<deploy>/volume.html?scene=/sample_scenes/voronoi.json
```

The browser fetches the scene JSON once, builds the voxel buffer geometry, and
plays back the precomputed time frames with the timeline scrubber.

## Live mode (Vercel viewer + local Python backend)

1. Run the live-sim server locally:
   ```bash
   python -m laser_sim sim-live --primitive zigzag --grid-n 24 --nz 12 --port 8765
   ```
2. Expose it publicly via a tunnel:
   ```bash
   cloudflared tunnel --url http://localhost:8765
   ```
3. Open the Vercel viewer pointing at the tunnel:
   ```
   https://<deploy>/volume.html?live=https://<tunnel>/api/stream
   ```

The viewer subscribes via `EventSource`, decodes the base64 + uint8 cubes, and
updates the voxel buffer in place per frame.

## Schema v1 vs v2

`schema_version` field in scene.json distinguishes:

- **v1**: 2D `t_max_K[Nx][Ny]` arrays, no `grid_z_mm` — open `viewer.html`
- **v2**: 3D `t_max_K[Nx][Ny][Nz]` arrays + `grid_z_mm` + `depth_mm` — open `volume.html`

The volume viewer detects v1 inputs and renders a redirect hint to `viewer.html`.
