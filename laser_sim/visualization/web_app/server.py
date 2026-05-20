"""Flask-based read-only dashboard for laser_sim campaigns + 3D live sim.

Routes:
  GET /                      campaigns index + Pareto archive viewer
  GET /campaign/<name>       list candidates + Pareto archive of one campaign
  GET /api/campaigns         JSON list of campaigns
  GET /api/campaign/<name>   JSON archive of one campaign
  GET /api/calibrations/<scenario_id>  latest calibration posterior
  GET /viewer                3D live sim viewer (Three.js HTML page)
  GET /scene.json            current scene (set via ?path=...) or last exported
  GET /static/<path>         passthrough for the Three.js HTML

The viewer is purely browser-side; the server just provides data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flask import Flask, Response, abort, jsonify, send_file, send_from_directory

from laser_sim.control_plane.events import EventBus
from laser_sim.storage import open_database

_THREE_JS_DIR = Path(__file__).resolve().parents[1] / "three_js_app"
_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


_INDEX_HTML = """<!doctype html>
<html><head>
<meta charset="utf-8">
<title>laser_sim · dashboard</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0a0a0a; color: #ddd; margin: 0; padding: 24px;
         font-size: 14px; }
  h1 { font-weight: 500; }
  table { border-collapse: collapse; width: 100%; margin: 12px 0 24px; }
  th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid #2a2a2a; }
  th { color: #9ad; background: #161616; }
  a { color: #6cf; text-decoration: none; }
  a:hover { text-decoration: underline; }
  .num { font-family: ui-monospace, "SF Mono", Menlo, monospace; color: #fff; }
  .head { display: flex; gap: 18px; align-items: baseline; margin-bottom: 18px; }
  .head a { background: #2a4d8a; color: white; padding: 6px 12px; border-radius: 4px; }
  .head a:hover { background: #355fa9; text-decoration: none; }
</style>
</head><body>
<div class="head">
  <h1 style="margin:0;">laser_sim · dashboard</h1>
  <a href="/viewer">3D live sim viewer</a>
</div>
<h2>Campaigns</h2>
<table>
  <tr><th>name</th><th>scenario</th><th>gens</th><th>archive</th><th>updated</th></tr>
__CAMPAIGN_ROWS__
</table>
__CAMPAIGN_DETAIL__
</body></html>
"""


_CAMPAIGN_DETAIL_HTML = """
<h2>Campaign: __CNAME__ · archive (top 30)</h2>
<table>
  <tr><th>scalar J</th><th>fitness</th><th>gen</th></tr>
__ARCHIVE_ROWS__
</table>
"""


def create_app(
    db_path: Path,
    scene_json: Path | None = None,
    event_bus: EventBus | None = None,
) -> Flask:
    app = Flask(__name__, static_folder=str(_THREE_JS_DIR), static_url_path="/static")
    db_path = Path(db_path)
    scene_path = Path(scene_json) if scene_json else None

    def _render_index(detail_html: str = "") -> str:
        if not db_path.exists():
            return f"<p>no database at {db_path}; run <code>cli evolve --persist {db_path}</code> first.</p>"
        db = open_database(db_path)
        try:
            campaigns = db.list_campaigns()
            rows: list[str] = []
            for c in campaigns:
                arc = db.list_archive(c.campaign_id)
                rows.append(
                    f"<tr><td><a href='/campaign/{c.name}'>{c.name}</a></td>"
                    f"<td class='num'>{c.scenario_id[:8]}</td>"
                    f"<td class='num'>{c.last_generation}</td>"
                    f"<td class='num'>{len(arc)}</td>"
                    f"<td class='num'>{c.updated_at[:19]}</td></tr>"
                )
        finally:
            db.close()
        return (
            _INDEX_HTML.replace("__CAMPAIGN_ROWS__", "\n".join(rows) or "<tr><td colspan=5>no campaigns</td></tr>")
            .replace("__CAMPAIGN_DETAIL__", detail_html)
        )

    @app.route("/")
    def index():
        return _render_index()

    @app.route("/campaign/<name>")
    def campaign(name):
        if not db_path.exists():
            abort(404)
        db = open_database(db_path)
        try:
            row = db.get_campaign_by_name(name)
            if row is None:
                abort(404)
            archive = db.list_archive(row.campaign_id)
        finally:
            db.close()
        rows: list[str] = []
        for a in archive[:30]:
            sj = f"{a.scalar_J:.4f}" if a.scalar_J is not None else "-"
            rows.append(
                f"<tr><td class='num'>{sj}</td>"
                f"<td class='num'>{a.fitness_json}</td>"
                f"<td class='num'>{a.generation if a.generation is not None else '-'}</td></tr>"
            )
        detail = _CAMPAIGN_DETAIL_HTML.replace("__CNAME__", name).replace(
            "__ARCHIVE_ROWS__", "\n".join(rows) or "<tr><td colspan=3>empty</td></tr>"
        )
        return _render_index(detail)

    @app.route("/api/campaigns")
    def api_campaigns():
        if not db_path.exists():
            return jsonify([])
        db = open_database(db_path)
        try:
            return jsonify(
                [
                    {
                        "name": c.name,
                        "campaign_id": c.campaign_id,
                        "scenario_id": c.scenario_id,
                        "last_generation": c.last_generation,
                        "updated_at": c.updated_at,
                    }
                    for c in db.list_campaigns()
                ]
            )
        finally:
            db.close()

    @app.route("/api/campaign/<name>")
    def api_campaign(name):
        if not db_path.exists():
            abort(404)
        db = open_database(db_path)
        try:
            row = db.get_campaign_by_name(name)
            if row is None:
                abort(404)
            return jsonify(
                {
                    "name": row.name,
                    "campaign_id": row.campaign_id,
                    "scenario_id": row.scenario_id,
                    "last_generation": row.last_generation,
                    "archive": [
                        {
                            "candidate_id": a.candidate_id,
                            "fitness": json.loads(a.fitness_json),
                            "scalar_J": a.scalar_J,
                            "generation": a.generation,
                        }
                        for a in db.list_archive(row.campaign_id)
                    ],
                }
            )
        finally:
            db.close()

    @app.route("/api/calibrations/<scenario_id>")
    def api_calibration(scenario_id):
        if not db_path.exists():
            abort(404)
        db = open_database(db_path)
        try:
            return jsonify(db.latest_calibration(scenario_id) or {})
        finally:
            db.close()

    @app.route("/viewer")
    def viewer():
        return send_from_directory(_THREE_JS_DIR, "index.html")

    @app.route("/volume")
    def volume_viewer():
        return send_from_directory(_THREE_JS_DIR, "volume.html")

    @app.route("/live")
    def live():
        return send_from_directory(_THREE_JS_DIR, "live.html")

    @app.route("/api/stream")
    def api_stream():
        if event_bus is None:
            abort(404)

        def _gen():
            q, backlog = event_bus.subscribe()
            try:
                # send historical events first so a late-joining browser
                # sees prior generations
                for ev in backlog:
                    yield ev.to_sse()
                while True:
                    try:
                        ev = q.get(timeout=15)
                        yield ev.to_sse()
                    except Exception:
                        # keep-alive ping so proxies don't time us out
                        yield ": keep-alive\n\n"
            finally:
                event_bus.unsubscribe(q)

        return Response(
            _gen(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    @app.route("/scene.json")
    def scene_serve():
        if scene_path is None or not scene_path.exists():
            abort(404)
        return send_file(str(scene_path), mimetype="application/json")

    return app


def serve(
    db_path: Path = Path("laser_sim_campaigns.db"),
    scene_json: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    debug: bool = False,
    event_bus: EventBus | None = None,
) -> None:
    app = create_app(db_path=db_path, scene_json=scene_json, event_bus=event_bus)
    app.run(host=host, port=port, debug=debug, threaded=True)
