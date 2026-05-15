"""Command-line entry point.

Subcommands grow with the phases. P0/P2 cover `info`, `scenario`, and
`viz-pattern`. Later phases add `fast-sim`, `pack-bed`, `evolve`, `validate`,
`closed-loop`.
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


if __name__ == "__main__":
    app()
