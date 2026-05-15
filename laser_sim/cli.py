"""Command-line entry point.

Subcommands grow with the phases. P0/P2 cover `info`, `scenario`, and
`viz-pattern`. P3 (proxy) + P4 add `evolve`. Later phases add `fast-sim`
(real solver), `pack-bed`, `validate`, `closed-loop`.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
import yaml
from rich.console import Console
from rich.table import Table

from laser_sim import __version__
from laser_sim.config.schema import ScenarioConfig
from laser_sim.patterns.base import PrimitiveKind, PrimitiveSpec
from laser_sim.patterns.primitives import build_primitive, list_primitives
from laser_sim.visualization.pattern_plot import plot_pattern

app = typer.Typer(
    name="laser-sim",
    help="Multi-fidelity evolutionary optimizer for LPBF laser scan patterns.",
    no_args_is_help=True,
)
scenario_app = typer.Typer(help="Scenario management (P0 scope/schema).")
app.add_typer(scenario_app, name="scenario")

_console = Console()

_DEFAULT_SCENARIO = Path(__file__).parent / "config" / "default.yaml"


def _load_scenario(path: Path | None) -> ScenarioConfig:
    p = path or _DEFAULT_SCENARIO
    with open(p) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig.model_validate(data)


@app.command()
def info() -> None:
    """Print package info and registered primitives."""
    table = Table(title=f"laser-sim {__version__}")
    table.add_column("key")
    table.add_column("value")
    table.add_row("version", __version__)
    table.add_row("primitives", ", ".join(list_primitives()))
    table.add_row("default scenario", str(_DEFAULT_SCENARIO))
    _console.print(table)


@scenario_app.command("show")
def scenario_show(
    path: Path = typer.Option(None, "--path", "-p", help="Scenario YAML path"),
) -> None:
    """Validate and pretty-print a scenario YAML."""
    sc = _load_scenario(path)
    _console.print_json(sc.model_dump_json(indent=2))


@scenario_app.command("validate")
def scenario_validate(
    path: Path = typer.Option(..., "--path", "-p"),
) -> None:
    """Validate a scenario YAML against the schema. Exit code != 0 on failure."""
    try:
        sc = _load_scenario(path)
    except Exception as e:
        _console.print(f"[red]invalid:[/red] {e}")
        raise typer.Exit(code=2)
    _console.print(f"[green]ok[/green] scenario {sc.scenario_id}")


@app.command("viz-pattern")
def viz_pattern(
    primitive: str = typer.Option("zigzag", "--primitive", help=f"one of: {list_primitives()}"),
    power: float = typer.Option(200.0, "--power", help="W"),
    speed: float = typer.Option(800.0, "--speed", help="mm/s"),
    hatch: float = typer.Option(100.0, "--hatch", help="um"),
    spot: float = typer.Option(80.0, "--spot", help="um"),
    rotation: float = typer.Option(0.0, "--rotation", help="deg"),
    stripe_width: float = typer.Option(5.0, "--stripe-width", help="mm (stripes only)"),
    samples_per_turn: int = typer.Option(64, "--samples-per-turn", help="spiral only"),
    scenario: Path = typer.Option(None, "--scenario", help="scenario YAML (defaults to bundled)"),
    out: Path = typer.Option(Path("pattern.png"), "--out", "-o", help="output PNG"),
    json_out: Path = typer.Option(None, "--json-out", help="optional pattern JSON dump"),
) -> None:
    """Render a primitive into the scenario ROI and save a PNG preview."""
    try:
        kind = PrimitiveKind(primitive)
    except ValueError:
        _console.print(f"[red]unknown primitive[/red]: {primitive}")
        _console.print(f"available: {list_primitives()}")
        raise typer.Exit(code=2)
    sc = _load_scenario(scenario)
    spec = PrimitiveSpec(
        kind=kind,
        power_W=power,
        speed_mm_s=speed,
        hatch_um=hatch,
        spot_um=spot,
        rotation_deg=rotation,
        params={"stripe_width_mm": stripe_width, "samples_per_turn": samples_per_turn},
    )
    try:
        pattern = build_primitive(spec, sc.roi)
    except NotImplementedError as e:
        _console.print(f"[yellow]not implemented:[/yellow] {e}")
        raise typer.Exit(code=2)
    saved = plot_pattern(
        pattern,
        out=out,
        title=f"{kind.value} | P={power:.0f}W v={speed:.0f}mm/s hatch={hatch:.0f}um",
    )
    _console.print(f"[green]wrote[/green] {saved}")
    _console.print(
        f"segments={len(pattern.segments)} "
        f"length={pattern.total_length_mm():.2f}mm "
        f"time={pattern.total_time_s()*1e3:.1f}ms"
    )
    if json_out is not None:
        payload = {
            "kind": kind.value,
            "spec": {
                "power_W": power,
                "speed_mm_s": speed,
                "hatch_um": hatch,
                "spot_um": spot,
                "rotation_deg": rotation,
            },
            "scenario_id": sc.scenario_id,
            "objective_version": sc.objective_version,
            "n_segments": len(pattern.segments),
            "total_length_mm": pattern.total_length_mm(),
            "total_time_s": pattern.total_time_s(),
        }
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(payload, indent=2))
        _console.print(f"[green]wrote[/green] {json_out}")


@app.command("viz-field")
def viz_field(
    primitive: str = typer.Option("zigzag", "--primitive"),
    power: float = typer.Option(200.0, "--power", help="W"),
    speed: float = typer.Option(800.0, "--speed", help="mm/s"),
    hatch: float = typer.Option(100.0, "--hatch", help="um"),
    spot: float = typer.Option(80.0, "--spot", help="um"),
    rotation: float = typer.Option(0.0, "--rotation", help="deg"),
    grid_n: int = typer.Option(41, "--grid-n"),
    stride: int = typer.Option(4, "--stride", help="path subsampling for the field eval"),
    scenario: Path = typer.Option(None, "--scenario"),
    out: Path = typer.Option(Path("field.png"), "--out", "-o"),
) -> None:
    """Render the T_max field for a primitive over the scenario ROI."""
    from laser_sim.fitness.evaluator import PatternEvaluator
    from laser_sim.patterns.rasterize import rasterize_pattern
    from laser_sim.visualization.field_plot import plot_field

    try:
        kind = PrimitiveKind(primitive)
    except ValueError:
        _console.print(f"[red]unknown primitive[/red]: {primitive}")
        raise typer.Exit(code=2)
    sc = _load_scenario(scenario)
    spec = PrimitiveSpec(
        kind=kind,
        power_W=power,
        speed_mm_s=speed,
        hatch_um=hatch,
        spot_um=spot,
        rotation_deg=rotation,
    )
    pat = build_primitive(spec, sc.roi)
    ev = PatternEvaluator(
        sc.material, sc.machine, sc.roi, mode="field", grid_n=grid_n, stride=stride, spot_um=spot
    )
    out_eval = ev.evaluate(pat, hatch_mm=hatch * 1e-3, spot_um=spot)
    if out_eval.field is None:
        _console.print("[red]no field result[/red]")
        raise typer.Exit(code=1)
    rp = rasterize_pattern(pat, ds_mm=0.05)
    saved = plot_field(
        out_eval.field,
        path=rp,
        out=out,
        title=(
            f"{kind.value} | P={power:.0f}W v={speed:.0f}mm/s "
            f"hatch={hatch:.0f}um spot={spot:.0f}um"
        ),
    )
    _console.print(
        f"[green]wrote[/green] {saved} | coverage={out_eval.field.coverage_fraction*100:.1f}% "
        f"keyhole={out_eval.field.keyhole_fraction*100:.1f}%"
    )


@app.command()
def evolve(
    population: int = typer.Option(16, "--pop", help="population size"),
    generations: int = typer.Option(5, "--gens", help="number of generations"),
    seed: int = typer.Option(42, "--seed"),
    elitism: int = typer.Option(2, "--elitism"),
    crossover_rate: float = typer.Option(0.7, "--crossover"),
    mutation_rate: float = typer.Option(0.3, "--mutation"),
    scenario: Path = typer.Option(None, "--scenario", help="scenario YAML"),
    archive_json: Path = typer.Option(None, "--archive-json", help="dump Pareto archive"),
    best_pattern_png: Path = typer.Option(
        None, "--best-pattern-png", help="render the best (scalar) pattern"
    ),
    best_field_png: Path = typer.Option(
        None, "--best-field-png", help="render the best pattern's T_max field"
    ),
    mode: str = typer.Option("field", "--mode", help="evaluation mode: field|segment"),
    grid_n: int = typer.Option(41, "--grid-n"),
    stride: int = typer.Option(4, "--stride"),
    transient_mode: str = typer.Option(
        "max",
        "--transient",
        help="max (fast, default for EA) | superposition (true heat accumulation, ~5x slower)",
    ),
    persist_db: Path = typer.Option(
        None, "--persist", help="SQLite path to write campaign + runs + archive"
    ),
    resume_campaign: str = typer.Option(
        None,
        "--resume",
        help="campaign name to resume; reads cached evaluations from --persist DB",
    ),
    campaign_name: str = typer.Option(
        None, "--name", help="campaign name when persisting; auto-generated if omitted"
    ),
) -> None:
    """Run the NSGA-II evolutionary loop with the Eagar-Tsai proxy.

    This is the Phase 4 EA core wired to the Phase 3 analytical proxy. The
    JAX/CuPy fast-sim and the HF (laserbeamFoam) runner will plug in behind
    PatternEvaluator without changing this command.
    """
    from laser_sim.fitness.evaluator import OBJECTIVE_NAMES, PatternEvaluator
    from laser_sim.genome.engine import GeneticEngine

    sc = _load_scenario(scenario)
    ea = sc.ea.model_copy(
        update={
            "population": population,
            "generations": generations,
            "seed": seed,
            "elitism": elitism,
            "crossover_rate": crossover_rate,
            "mutation_rate": mutation_rate,
        }
    )
    _console.print(
        f"[bold]scenario[/bold] material={sc.material.material_id} "
        f"machine={sc.machine.machine_id} roi={sc.roi.roi_id} "
        f"pop={ea.population} gens={ea.generations} seed={ea.seed}"
    )
    if mode not in ("field", "segment"):
        _console.print(f"[red]unknown mode[/red]: {mode}")
        raise typer.Exit(code=2)
    evaluator = PatternEvaluator(
        sc.material,
        sc.machine,
        sc.roi,
        mode=mode,
        grid_n=grid_n,
        stride=stride,
        transient_mode=transient_mode,
    )

    accumulator = None
    db_handle = None
    if persist_db is not None or resume_campaign is not None:
        from laser_sim.storage import KnowledgeAccumulator, open_database
        import time as _time

        db_path = persist_db or Path("laser_sim_campaigns.db")
        db_handle = open_database(db_path)
        scenario_dict = json.loads(sc.model_dump_json())
        scenario_id = db_handle.upsert_scenario(scenario_dict)
        if resume_campaign:
            row = db_handle.get_campaign_by_name(resume_campaign)
            if row is None:
                _console.print(f"[red]campaign not found: {resume_campaign}[/red]")
                raise typer.Exit(code=2)
            cid = row.campaign_id
            _console.print(
                f"[bold]resuming[/bold] campaign {resume_campaign!r} "
                f"(id={cid[:8]}, last_gen={row.last_generation})"
            )
        else:
            cname = campaign_name or f"campaign_{int(_time.time())}"
            cid = db_handle.create_campaign(name=cname, scenario_id=scenario_id)
            _console.print(f"[bold]new campaign[/bold] {cname!r} (id={cid[:8]})")
        accumulator = KnowledgeAccumulator(db=db_handle, campaign_id=cid)
        if len(accumulator) > 0:
            _console.print(
                f"[green]hydrated[/green] {len(accumulator)} prior evaluations from disk"
            )

    def _commit(g: int, entries: list) -> None:
        if accumulator is not None:
            accumulator.commit_archive(entries, generation=g)

    table = Table(title="Evolution log")
    for col in ("gen", "front0", "archive", "best_J", "median_J", "HV2D", "crisis"):
        table.add_column(col)

    rows: list[tuple] = []

    def _on_gen(stats) -> None:
        rows.append(
            (
                str(stats.generation),
                str(stats.front0_size),
                str(stats.archive_size),
                f"{stats.best_scalar_J:.4f}",
                f"{stats.median_scalar_J:.4f}",
                f"{stats.hypervolume_2d:.4f}",
                "*" if stats.crisis else "",
            )
        )

    eng = GeneticEngine(
        material=sc.material,
        machine=sc.machine,
        roi=sc.roi,
        ea=ea,
        evaluator=evaluator,
        on_generation=_on_gen,
        cache=accumulator,
        on_commit=_commit,
    )
    log = eng.run()
    if db_handle is not None:
        db_handle.close()
    for r in rows:
        table.add_row(*r)
    _console.print(table)

    best = log.best()
    if best is None:
        _console.print("[red]no solutions in archive[/red]")
        raise typer.Exit(code=1)
    _console.print(
        "[bold green]Pareto archive[/bold green] "
        f"size={len(log.archive)} best_J={best.scalar_J:.4f}"
    )
    _console.print("best chromosome:")
    _console.print_json(
        json.dumps(
            {
                "kind": best.chromosome.primitive_kind.value,
                "power_W": round(best.chromosome.power_W, 3),
                "speed_mm_s": round(best.chromosome.speed_mm_s, 3),
                "hatch_um": round(best.chromosome.hatch_um, 3),
                "spot_um": round(best.chromosome.spot_um, 3),
                "rotation_deg": best.chromosome.layer_rotation_deg,
                "extras": best.chromosome.extras,
            }
        )
    )

    if archive_json is not None:
        payload = {
            "scenario_id": sc.scenario_id,
            "objective_version": sc.objective_version,
            "objectives": list(OBJECTIVE_NAMES),
            "archive": [
                {
                    "fitness": list(e.fitness.values),
                    "scalar_J": e.scalar_J,
                    "chromosome": {
                        "kind": e.chromosome.primitive_kind.value,
                        "power_W": e.chromosome.power_W,
                        "speed_mm_s": e.chromosome.speed_mm_s,
                        "hatch_um": e.chromosome.hatch_um,
                        "spot_um": e.chromosome.spot_um,
                        "rotation_deg": e.chromosome.layer_rotation_deg,
                        "extras": e.chromosome.extras,
                    },
                }
                for e in log.archive.entries
            ],
        }
        archive_json.parent.mkdir(parents=True, exist_ok=True)
        archive_json.write_text(json.dumps(payload, indent=2))
        _console.print(f"[green]wrote[/green] {archive_json}")

    if best_pattern_png is not None or best_field_png is not None:
        pat = build_primitive(best.chromosome.to_primitive_spec(), sc.roi)
        title = (
            f"best | {best.chromosome.primitive_kind.value} | "
            f"P={best.chromosome.power_W:.0f}W "
            f"v={best.chromosome.speed_mm_s:.0f}mm/s "
            f"hatch={best.chromosome.hatch_um:.0f}um"
        )
        if best_pattern_png is not None:
            saved = plot_pattern(pat, out=best_pattern_png, title=title)
            _console.print(f"[green]wrote[/green] {saved}")
        if best_field_png is not None:
            from laser_sim.patterns.rasterize import rasterize_pattern
            from laser_sim.visualization.field_plot import plot_field

            ev_best = evaluator.evaluate(
                pat, hatch_mm=best.chromosome.hatch_um * 1e-3, spot_um=best.chromosome.spot_um
            )
            if ev_best.field is None:
                _console.print(
                    "[yellow]field render skipped[/yellow]: evaluator running in segment mode"
                )
            else:
                rp = rasterize_pattern(pat, ds_mm=0.05)
                saved = plot_field(ev_best.field, path=rp, out=best_field_png, title=title)
                _console.print(f"[green]wrote[/green] {saved}")


@app.command("validate")
def validate(
    sim_pattern_kind: str = typer.Option(
        "zigzag",
        "--sim-primitive",
        help="primitive to use for the simulated reference (sim ↔ real comparison)",
    ),
    sim_power: float = typer.Option(200.0, "--sim-power"),
    sim_speed: float = typer.Option(800.0, "--sim-speed"),
    sim_hatch: float = typer.Option(100.0, "--sim-hatch"),
    sim_spot: float = typer.Option(80.0, "--sim-spot"),
    real_dir: Path = typer.Option(
        ..., "--real", help="directory containing thermal*.npz, pyrometer*.csv, metadata.json"
    ),
    scenario: Path = typer.Option(None, "--scenario"),
    grid_n: int = typer.Option(41, "--grid-n"),
    stride: int = typer.Option(4, "--stride"),
    report_png: Path = typer.Option(None, "--report-png", help="side-by-side sim/real heatmap"),
) -> None:
    """Compare a simulated pattern to a measurement set on disk.

    Computes RMSE/IoU/cosine/KL between the sim T_max field and the
    ingested real thermal frame after rigid-2D registration onto the sim grid.
    The real measurement set must follow the canonical layout documented in
    laser_sim/validation/ingest.py.
    """
    from laser_sim.fitness.evaluator import PatternEvaluator
    from laser_sim.validation import (
        FieldFeatures,
        extract_field_features,
        load_measurement_set,
        score_thermal_frame,
    )
    from laser_sim.validation.register import resample_field_to_grid

    sc = _load_scenario(scenario)
    try:
        kind = PrimitiveKind(sim_pattern_kind)
    except ValueError:
        _console.print(f"[red]unknown primitive[/red]: {sim_pattern_kind}")
        raise typer.Exit(code=2)
    spec = PrimitiveSpec(
        kind=kind,
        power_W=sim_power,
        speed_mm_s=sim_speed,
        hatch_um=sim_hatch,
        spot_um=sim_spot,
    )
    pat = build_primitive(spec, sc.roi)
    ev = PatternEvaluator(
        sc.material, sc.machine, sc.roi, mode="field", grid_n=grid_n, stride=stride, spot_um=sim_spot
    )
    sim_eval = ev.evaluate(pat, hatch_mm=sim_hatch * 1e-3, spot_um=sim_spot)
    if sim_eval.field is None:
        _console.print("[red]sim eval missing field[/red]")
        raise typer.Exit(code=1)

    measurement_set = load_measurement_set(real_dir)
    real_thermal = measurement_set.first_thermal()
    if real_thermal is None:
        _console.print(f"[red]no thermal*.npz in {real_dir}[/red]")
        raise typer.Exit(code=2)
    _console.print(
        f"loaded {len(measurement_set.items)} measurements; experiment_id="
        f"{measurement_set.experiment_id} sensor={real_thermal.sensor}"
    )

    # construct grid from real frame's pixel size + origin
    H, W = real_thermal.frame_K.shape
    real_x = real_thermal.origin_xy_mm[0] + np.arange(H) * real_thermal.pixel_um * 1e-3
    real_y = real_thermal.origin_xy_mm[1] + np.arange(W) * real_thermal.pixel_um * 1e-3
    # resample real onto sim grid
    real_on_sim = resample_field_to_grid(
        real_thermal.frame_K, real_x, real_y, sim_eval.field.grid_x_mm, sim_eval.field.grid_y_mm
    )
    # mask both fields where real has NaN (out-of-coverage)
    valid = np.isfinite(real_on_sim)
    if not valid.any():
        _console.print("[red]no overlap between real and sim grids[/red]")
        raise typer.Exit(code=2)
    sim_masked = np.where(valid, sim_eval.field.t_max_K, np.nan)

    score = score_thermal_frame(
        sim_field_K=sim_masked,
        real_field_K=real_on_sim,
        liquidus_K=sc.material.liquidus_K,
        boiling_K=sc.material.boiling_K,
    )
    table = Table(title=f"Sim ↔ real validation ({measurement_set.experiment_id})")
    for col in ("metric", "value", "interpretation"):
        table.add_column(col)
    table.add_row("RMSE [K]", f"{score.rmse_K:.1f}", "lower is better")
    table.add_row("IoU melt", f"{score.iou_melt:.3f}", "1.0 = perfect overlap")
    table.add_row("IoU keyhole", f"{score.iou_keyhole:.3f}", "1.0 = perfect overlap")
    table.add_row("cosine(hist)", f"{score.cosine_hist:.3f}", "1.0 = identical T distribution")
    table.add_row("KL(sim||real)", f"{score.kl_hist:.4f}", "0.0 = identical T distribution")
    table.add_row("composite", f"{score.composite:.4f}", "lower is better")
    _console.print(table)

    if report_png is not None:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        vmin = float(np.nanmin([sim_masked, real_on_sim]))
        vmax = float(np.nanmax([sim_masked, real_on_sim]))
        extent = (
            float(sim_eval.field.grid_x_mm[0]),
            float(sim_eval.field.grid_x_mm[-1]),
            float(sim_eval.field.grid_y_mm[0]),
            float(sim_eval.field.grid_y_mm[-1]),
        )
        for ax, data, label in zip(
            axes,
            (sim_masked.T, real_on_sim.T, (sim_masked - real_on_sim).T),
            ("Simulation", f"Real ({real_thermal.sensor})", "Diff"),
        ):
            kw = dict(origin="lower", extent=extent, aspect="equal")
            if label == "Diff":
                im = ax.imshow(data, cmap="seismic", vmin=-(vmax - vmin) / 2, vmax=(vmax - vmin) / 2, **kw)
            else:
                im = ax.imshow(data, cmap="inferno", vmin=vmin, vmax=vmax, **kw)
            fig.colorbar(im, ax=ax, label="[K]")
            ax.set_title(label)
            ax.set_xlabel("x [mm]")
            ax.set_ylabel("y [mm]")
        fig.suptitle(
            f"Validation | RMSE={score.rmse_K:.0f}K IoU_melt={score.iou_melt:.2f} "
            f"composite={score.composite:.3f}"
        )
        fig.tight_layout()
        report_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_png, dpi=140)
        plt.close(fig)
        _console.print(f"[green]wrote[/green] {report_png}")


@app.command("viz3d")
def viz3d(
    primitive: str = typer.Option("zigzag", "--primitive"),
    power: float = typer.Option(200.0, "--power"),
    speed: float = typer.Option(800.0, "--speed"),
    hatch: float = typer.Option(100.0, "--hatch"),
    spot: float = typer.Option(80.0, "--spot"),
    rotation: float = typer.Option(0.0, "--rotation"),
    scenario: Path = typer.Option(None, "--scenario"),
    grid_n: int = typer.Option(32, "--grid-n"),
    n_frames: int = typer.Option(24, "--n-frames"),
    stride: int = typer.Option(6, "--stride"),
    transient_mode: str = typer.Option(
        "superposition",
        "--transient",
        help="max | superposition (default: physical 3D Green's function)",
    ),
    out_json: Path = typer.Option(Path("scene.json"), "--out", "-o"),
) -> None:
    """Export a scan-pattern + time-stepped T_max field as a JSON scene that
    the Three.js viewer (laser_sim/visualization/three_js_app/index.html)
    consumes. Run `python -m laser_sim serve --scene scene.json` to view it
    at http://127.0.0.1:8765/viewer."""
    from laser_sim.visualization.scene_export import export_scene

    sc = _load_scenario(scenario)
    try:
        kind = PrimitiveKind(primitive)
    except ValueError:
        _console.print(f"[red]unknown primitive[/red]: {primitive}")
        raise typer.Exit(code=2)
    spec = PrimitiveSpec(
        kind=kind, power_W=power, speed_mm_s=speed, hatch_um=hatch, spot_um=spot, rotation_deg=rotation
    )
    pat = build_primitive(spec, sc.roi)
    saved = export_scene(
        pat,
        sc,
        out_path=out_json,
        spot_um=spot,
        grid_n=grid_n,
        n_frames=n_frames,
        stride=stride,
        transient_mode=transient_mode,
    )
    _console.print(
        f"[green]wrote[/green] {saved} ({n_frames} frames, {grid_n}^2 grid, "
        f"{pat.total_time_s()*1e3:.0f}ms scan, transient={transient_mode})"
    )
    _console.print(
        "next: [bold]python -m laser_sim serve --scene "
        f"{saved}[/bold] then open http://127.0.0.1:8765/viewer"
    )


@app.command("serve")
def serve_cmd(
    db_path: Path = typer.Option(
        Path("laser_sim_campaigns.db"), "--db", help="campaign database"
    ),
    scene: Path = typer.Option(None, "--scene", help="3D scene JSON to serve at /scene.json"),
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8765, "--port"),
) -> None:
    """Run the Flask dashboard + 3D viewer locally."""
    from laser_sim.visualization.web_app import serve as _serve

    _console.print(f"[bold]serving[/bold] http://{host}:{port}/  (Ctrl+C to stop)")
    _console.print(f"  db:    {db_path}")
    _console.print(f"  scene: {scene}")
    _serve(db_path=db_path, scene_json=scene, host=host, port=port, debug=False)


@app.command("history")
def history(
    db_path: Path = typer.Option(Path("laser_sim_campaigns.db"), "--db"),
    campaign: str = typer.Option(None, "--campaign", help="show one campaign's archive"),
    top: int = typer.Option(10, "--top"),
) -> None:
    """List campaigns or dump the Pareto archive of a given campaign."""
    from laser_sim.storage import open_database

    db = open_database(db_path)
    try:
        if campaign is None:
            campaigns = db.list_campaigns()
            table = Table(title=f"Campaigns in {db_path}")
            for c in ("name", "id", "scenario_id", "gens", "updated_at"):
                table.add_column(c)
            for r in campaigns:
                table.add_row(
                    r.name,
                    r.campaign_id[:8],
                    r.scenario_id[:8],
                    str(r.last_generation),
                    r.updated_at[:19],
                )
            _console.print(table)
            return
        row = db.get_campaign_by_name(campaign)
        if row is None:
            _console.print(f"[red]campaign not found[/red]: {campaign}")
            raise typer.Exit(code=2)
        archive = db.list_archive(row.campaign_id)
        _console.print(f"campaign [bold]{campaign}[/bold] archive size={len(archive)}")
        table = Table(title=f"Top {top} by scalar_J")
        for c in ("scalar_J", "fitness", "gen"):
            table.add_column(c)
        for a in archive[:top]:
            table.add_row(
                f"{a.scalar_J:.4f}" if a.scalar_J is not None else "-",
                a.fitness_json,
                str(a.generation) if a.generation is not None else "-",
            )
        _console.print(table)
    finally:
        db.close()


@app.command("calibrate")
def calibrate_cmd(
    sim_pattern_kind: str = typer.Option("zigzag", "--sim-primitive"),
    sim_power: float = typer.Option(200.0, "--sim-power"),
    sim_speed: float = typer.Option(800.0, "--sim-speed"),
    sim_hatch: float = typer.Option(100.0, "--sim-hatch"),
    real_dir: Path = typer.Option(..., "--real"),
    scenario: Path = typer.Option(None, "--scenario"),
    rounds: int = typer.Option(2, "--rounds"),
    out_json: Path = typer.Option(Path("posterior.json"), "--out"),
    grid_n: int = typer.Option(31, "--grid-n"),
    stride: int = typer.Option(6, "--stride"),
) -> None:
    """Calibrate (absorptivity, spot_um, emissivity) against a measurement set.

    Hierarchical 1D-line-search per parameter, repeated for `--rounds`. Writes
    a posterior JSON containing the optimum and the before/after composite
    scores. The optimum can be re-applied via `cli evolve` once
    persistence is wired (next slice).
    """
    from laser_sim.validation import (
        DEFAULT_LEVEL1_PARAMETERS,
        CalibrationProblem,
        calibrate,
        load_measurement_set,
    )

    sc = _load_scenario(scenario)
    try:
        kind = PrimitiveKind(sim_pattern_kind)
    except ValueError:
        _console.print(f"[red]unknown primitive[/red]: {sim_pattern_kind}")
        raise typer.Exit(code=2)
    spec = PrimitiveSpec(
        kind=kind,
        power_W=sim_power,
        speed_mm_s=sim_speed,
        hatch_um=sim_hatch,
        spot_um=80.0,
    )
    pat = build_primitive(spec, sc.roi)
    ms = load_measurement_set(real_dir)

    problem = CalibrationProblem(
        pattern=pat,
        measurements=ms,
        base_scenario=sc,
        parameters=DEFAULT_LEVEL1_PARAMETERS,
        hatch_mm=sim_hatch * 1e-3,
        grid_n=grid_n,
        stride=stride,
    )
    _console.print(
        f"[bold]calibrating[/bold] {len(DEFAULT_LEVEL1_PARAMETERS)} params over "
        f"{rounds} rounds against {ms.experiment_id} ({len(ms.items)} measurements)"
    )
    result = calibrate(problem, rounds=rounds)
    table = Table(title="Calibration result")
    for col in ("param", "default", "optimum", "delta"):
        table.add_column(col)
    for p in DEFAULT_LEVEL1_PARAMETERS:
        opt = result.optimum[p.name]
        table.add_row(p.name, f"{p.default:.3f}", f"{opt:.3f}", f"{opt - p.default:+.3f}")
    _console.print(table)
    _console.print(
        f"composite: [bold]{result.initial_score.composite:.4f}[/bold] -> "
        f"[bold green]{result.final_score.composite:.4f}[/bold green] "
        f"(rmse {result.initial_score.rmse_K:.0f}K -> {result.final_score.rmse_K:.0f}K)"
    )
    saved = result.to_json(out_json)
    _console.print(f"[green]wrote[/green] {saved}")


@app.command("viz-pareto")
def viz_pareto(
    archive_json: Path = typer.Option(..., "--archive", help="archive JSON from `evolve --archive-json`"),
    axes: str = typer.Option(
        "2,4",
        "--axes",
        help="objective indices to plot, comma-separated; 0=u_temp,1=lof,2=keyhole,3=surface,4=t_cycle",
    ),
    matrix: bool = typer.Option(False, "--matrix", help="render full pairwise scatter matrix"),
    out: Path = typer.Option(Path("pareto.png"), "--out", "-o"),
) -> None:
    """Render the Pareto archive (from `evolve --archive-json`) as a 2D scatter
    or full pairwise matrix."""
    from laser_sim.fitness.evaluator import OBJECTIVE_NAMES
    from laser_sim.visualization.pareto_plot import plot_pareto_2d, plot_pareto_matrix

    data = json.loads(Path(archive_json).read_text())
    archive = data["archive"]
    fits = [tuple(e["fitness"]) for e in archive]
    sca = [float(e["scalar_J"]) for e in archive]
    if matrix:
        saved = plot_pareto_matrix(fits, sca, objective_names=tuple(OBJECTIVE_NAMES), out=out)
    else:
        i, j = (int(x) for x in axes.split(","))
        saved = plot_pareto_2d(
            fits, sca, axes=(i, j), objective_names=tuple(OBJECTIVE_NAMES), out=out,
        )
    _console.print(f"[green]wrote[/green] {saved} ({len(fits)} archive members)")


@app.command("closed-loop")
def closed_loop(
    population: int = typer.Option(12, "--pop"),
    generations: int = typer.Option(3, "--gens"),
    seed: int = typer.Option(42, "--seed"),
    machine_kind: str = typer.Option(
        "mock", "--machine", help="mock | opcua (opcua not implemented in this slice)"
    ),
    machine_id: str = typer.Option(
        None,
        "--machine-id",
        help="target machine_id; required for non-mock machines, must be in allowlist",
    ),
    allowlist: str = typer.Option(
        "", "--allowlist", help="comma-separated allowlist of machine ids"
    ),
    i_know_what_im_doing: bool = typer.Option(
        False,
        "--i-know-what-im-doing",
        help="REQUIRED for non-mock machines; explicit human acknowledgement",
    ),
    time_scale: float = typer.Option(
        0.0,
        "--time-scale",
        help="MockMachine wall-clock dilation; 0=instant, 1=real-time",
    ),
    scenario: Path = typer.Option(None, "--scenario"),
    thermal_png: Path = typer.Option(None, "--thermal-png", help="dump synthetic thermal frame"),
) -> None:
    """Run a short EA campaign and deploy the best Pareto member to a machine
    (default: MockMachine) after passing every pattern through SafetyChecker.

    Real hardware is intentionally not implemented in this slice; the path is
    architecturally complete (Machine ABC + SafetyChecker + OpcUaConfig stub).
    """
    import asyncio

    from laser_sim.fitness.evaluator import PatternEvaluator
    from laser_sim.genome.engine import GeneticEngine
    from laser_sim.hardware import (
        MockMachine,
        SafetyChecker,
        SafetyViolation,
    )
    from laser_sim.hardware.safety import SafetyConfig

    sc = _load_scenario(scenario)
    ea = sc.ea.model_copy(
        update={
            "population": population,
            "generations": generations,
            "seed": seed,
        }
    )
    if machine_kind == "mock":
        machine = MockMachine(scenario=sc, time_scale=time_scale)
        target_id = sc.machine.machine_id
        is_mock = True
    elif machine_kind == "opcua":
        _console.print(
            "[red]opcua bridge not implemented in this slice[/red]: "
            "install with `pip install -e \".[hardware]\"` and wire a vendor adapter"
        )
        raise typer.Exit(code=2)
    else:
        _console.print(f"[red]unknown --machine[/red]: {machine_kind}")
        raise typer.Exit(code=2)

    allow = tuple(s for s in allowlist.split(",") if s.strip())
    safety = SafetyChecker(config=SafetyConfig(machine_allowlist=allow))
    _console.print(
        f"[bold]closed-loop[/bold] machine={machine_kind} id={target_id} "
        f"pop={ea.population} gens={ea.generations} ack={i_know_what_im_doing}"
    )

    evaluator = PatternEvaluator(sc.material, sc.machine, sc.roi, mode="field", grid_n=31, stride=6)
    eng = GeneticEngine(
        material=sc.material,
        machine=sc.machine,
        roi=sc.roi,
        ea=ea,
        evaluator=evaluator,
    )
    log = eng.run()
    best = log.best()
    if best is None:
        _console.print("[red]EA produced no solutions[/red]")
        raise typer.Exit(code=1)
    _console.print(
        f"[green]EA done[/green] archive={len(log.archive)} "
        f"best_J={best.scalar_J:.4f} kind={best.chromosome.primitive_kind.value}"
    )
    pattern = build_primitive(best.chromosome.to_primitive_spec(), sc.roi)

    # Safety gate
    decision = safety.check(
        pattern=pattern,
        scenario=sc,
        target_machine_id=target_id,
        acknowledge=i_know_what_im_doing,
        is_mock=is_mock,
        hatch_mm=best.chromosome.hatch_um * 1e-3,
    )
    if decision.warnings:
        for w in decision.warnings:
            _console.print(f"[yellow]safety warning[/yellow]: {w}")
    if not decision.allowed:
        for v in decision.violations:
            _console.print(f"[red]safety violation[/red]: {v}")
        try:
            decision.raise_if_blocked()
        except SafetyViolation as e:
            _console.print(f"[red]REJECTED[/red]: {e}")
            raise typer.Exit(code=3)
    _console.print("[green]safety OK[/green]")

    async def _run() -> None:
        job = await machine.upload_pattern(pattern)
        _console.print(f"uploaded job [bold]{job}[/bold]")
        await machine.start_build(job)
        _console.print("build started")
        st = await machine.status()
        _console.print(
            f"status: state={st.state.value} progress={st.progress*100:.1f}% "
            f"O2={st.chamber_o2_ppm:.0f}ppm"
        )
        thermal = await machine.read_thermal()
        if thermal is not None:
            _console.print(
                f"thermal frame: {thermal.frame_K.shape} K, pixel={thermal.pixel_um:.1f}um, "
                f"sensor={thermal.sensor}, t_max={float(thermal.frame_K.max()):.0f}K"
            )
            if thermal_png is not None:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 6))
                im = ax.imshow(
                    thermal.frame_K.T,
                    origin="lower",
                    extent=(
                        thermal.origin_xy_mm[0],
                        thermal.origin_xy_mm[0] + thermal.frame_K.shape[0] * thermal.pixel_um / 1000.0,
                        thermal.origin_xy_mm[1],
                        thermal.origin_xy_mm[1] + thermal.frame_K.shape[1] * thermal.pixel_um / 1000.0,
                    ),
                    cmap="inferno",
                    aspect="equal",
                )
                fig.colorbar(im, ax=ax, label="T_max [K]")
                ax.set_title(
                    f"synthetic thermal | {machine_kind} | job {job} | sensor {thermal.sensor}"
                )
                ax.set_xlabel("x [mm]")
                ax.set_ylabel("y [mm]")
                thermal_png.parent.mkdir(parents=True, exist_ok=True)
                fig.tight_layout()
                fig.savefig(thermal_png, dpi=150)
                plt.close(fig)
                _console.print(f"[green]wrote[/green] {thermal_png}")
        pyro = await machine.read_pyrometer()
        if pyro is not None:
            _console.print(
                f"pyrometer: T={pyro.temperature_K:.0f}K at t={pyro.timestamp_s:.3f}s"
            )

    asyncio.run(_run())


if __name__ == "__main__":
    app()
