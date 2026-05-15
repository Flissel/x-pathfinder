"""Command-line entry point.

Subcommands grow with the phases. P0/P2 cover `info`, `scenario`, and
`viz-pattern`. P3 (proxy) + P4 add `evolve`. Later phases add `fast-sim`
(real solver), `pack-bed`, `validate`, `closed-loop`.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    evaluator = PatternEvaluator(sc.material, sc.machine, sc.roi)

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
    )
    log = eng.run()
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

    if best_pattern_png is not None:
        pat = build_primitive(best.chromosome.to_primitive_spec(), sc.roi)
        saved = plot_pattern(
            pat,
            out=best_pattern_png,
            title=(
                f"best | {best.chromosome.primitive_kind.value} | "
                f"P={best.chromosome.power_W:.0f}W "
                f"v={best.chromosome.speed_mm_s:.0f}mm/s "
                f"hatch={best.chromosome.hatch_um:.0f}um"
            ),
        )
        _console.print(f"[green]wrote[/green] {saved}")


if __name__ == "__main__":
    app()
