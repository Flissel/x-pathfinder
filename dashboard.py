"""
Live dashboard for X Pathfinder.

Shows real-time stats, email feed, and domain breakdown.
Uses Flask + Server-Sent Events for live updates.
"""

import json
import time
import queue
import threading
import logging
from datetime import datetime
from flask import Flask, Response, jsonify, render_template_string

from .database import EmailDatabase

logger = logging.getLogger(__name__)

# Global event queue for SSE
event_queue = queue.Queue(maxsize=1000)


def push_event(event_type: str, data: dict):
    """Push an event to the dashboard (called from daemon)."""
    try:
        event_queue.put_nowait({
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })
    except queue.Full:
        pass


DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X Pathfinder Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff41;
            min-height: 100vh;
        }
        .header {
            background: #111;
            border-bottom: 1px solid #00ff41;
            padding: 15px 30px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 18px; letter-spacing: 2px; }
        .header .status {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .pulse {
            width: 10px; height: 10px;
            background: #00ff41;
            border-radius: 50%;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.3; }
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            padding: 20px 30px;
        }
        .stat-card {
            background: #111;
            border: 1px solid #1a3a1a;
            border-radius: 4px;
            padding: 20px;
            text-align: center;
        }
        .stat-card .value {
            font-size: 36px;
            font-weight: bold;
            color: #00ff41;
        }
        .stat-card .label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 5px;
        }
        .stat-card.highlight { border-color: #00ff41; }
        .stat-card.highlight .value { color: #fff; text-shadow: 0 0 10px #00ff41; }
        .panels {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 15px;
            padding: 0 30px 15px;
        }
        .panel {
            background: #111;
            border: 1px solid #1a3a1a;
            border-radius: 4px;
            overflow: hidden;
        }
        .panel-header {
            background: #1a1a1a;
            padding: 10px 15px;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #888;
            border-bottom: 1px solid #1a3a1a;
        }
        .email-feed {
            height: 500px;
            overflow-y: auto;
            padding: 5px 0;
        }
        .email-row {
            padding: 6px 15px;
            font-size: 13px;
            border-bottom: 1px solid #0a0a0a;
            display: flex;
            justify-content: space-between;
            animation: fadeIn 0.3s ease;
        }
        .email-row:hover { background: #1a1a1a; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-5px); } to { opacity: 1; } }
        .email-row .email { color: #00ff41; flex: 1; }
        .email-row .handle { color: #666; width: 150px; text-align: right; }
        .email-row .badge {
            display: inline-block;
            padding: 1px 6px;
            border-radius: 3px;
            font-size: 10px;
            margin-left: 8px;
        }
        .badge.smtp { background: #003300; color: #00ff41; border: 1px solid #00ff41; }
        .badge.mx { background: #1a1a00; color: #ffff00; border: 1px solid #333300; }
        .badge.fail { background: #1a0000; color: #ff4444; border: 1px solid #330000; }
        .domain-list { padding: 10px 15px; }
        .domain-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            font-size: 13px;
            border-bottom: 1px solid #1a1a1a;
        }
        .domain-row .bar {
            height: 4px;
            background: #00ff41;
            margin-top: 3px;
            border-radius: 2px;
            transition: width 0.5s ease;
        }
        .rate-display {
            padding: 15px;
            font-size: 13px;
        }
        .rate-display .row {
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            color: #888;
        }
        .rate-display .row .val { color: #00ff41; }
        .footer {
            padding: 10px 30px;
            color: #333;
            font-size: 11px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>X PATHFINDER</h1>
        <div class="status">
            <div class="pulse" id="pulse"></div>
            <span id="daemon-status">CONNECTING...</span>
        </div>
    </div>

    <div class="grid">
        <div class="stat-card highlight">
            <div class="value" id="smtp-count">-</div>
            <div class="label">SMTP Verified</div>
        </div>
        <div class="stat-card">
            <div class="value" id="mx-count">-</div>
            <div class="label">MX Verified</div>
        </div>
        <div class="stat-card">
            <div class="value" id="total-count">-</div>
            <div class="label">Total Emails</div>
        </div>
        <div class="stat-card">
            <div class="value" id="account-count">-</div>
            <div class="label">Accounts</div>
        </div>
    </div>

    <div class="panels">
        <div class="panel">
            <div class="panel-header">Live Email Feed</div>
            <div class="email-feed" id="feed"></div>
        </div>
        <div>
            <div class="panel" style="margin-bottom: 15px;">
                <div class="panel-header">Top Domains</div>
                <div class="domain-list" id="domains"></div>
            </div>
            <div class="panel">
                <div class="panel-header">Performance</div>
                <div class="rate-display" id="perf">
                    <div class="row"><span>Emails/min</span><span class="val" id="rate">-</span></div>
                    <div class="row"><span>Generation</span><span class="val" id="gen">-</span></div>
                    <div class="row"><span>Uptime</span><span class="val" id="uptime">-</span></div>
                    <div class="row"><span>SMTP Hit Rate</span><span class="val" id="hitrate">-</span></div>
                </div>
            </div>
        </div>
    </div>

    <div style="padding: 0 30px 15px;">
        <div class="panel">
            <div class="panel-header">Emails by Country</div>
            <div style="display:flex;flex-wrap:wrap;gap:10px;padding:15px;" id="countries"></div>
        </div>
    </div>

    <div class="footer">SAKANA-INSPIRED EVOLUTIONARY EMAIL DISCOVERY</div>

    <script>
        let startTime = Date.now();
        let emailCount = 0;
        let lastCount = 0;
        let lastTime = Date.now();

        function formatNum(n) {
            if (n >= 1000000) return (n/1000000).toFixed(1) + 'M';
            if (n >= 1000) return (n/1000).toFixed(1) + 'K';
            return n.toString();
        }

        function updateUptime() {
            let secs = Math.floor((Date.now() - startTime) / 1000);
            let h = Math.floor(secs / 3600);
            let m = Math.floor((secs % 3600) / 60);
            let s = secs % 60;
            document.getElementById('uptime').textContent =
                (h > 0 ? h + 'h ' : '') + m + 'm ' + s + 's';
        }
        setInterval(updateUptime, 1000);

        // Poll stats every 2 seconds
        async function pollStats() {
            try {
                const resp = await fetch('/api/stats');
                const data = await resp.json();

                document.getElementById('smtp-count').textContent = formatNum(data.emails_smtp_verified);
                document.getElementById('mx-count').textContent = formatNum(data.emails_mx_verified);
                document.getElementById('total-count').textContent = formatNum(data.emails_total);
                document.getElementById('account-count').textContent = formatNum(data.accounts);
                document.getElementById('daemon-status').textContent = 'RUNNING';
                document.getElementById('gen').textContent = data.generation || '-';

                // Rate calculation
                let now = Date.now();
                let dt = (now - lastTime) / 60000; // minutes
                if (dt > 0) {
                    let rate = Math.round((data.emails_total - lastCount) / dt);
                    if (rate > 0) document.getElementById('rate').textContent = rate;
                    lastCount = data.emails_total;
                    lastTime = now;
                }

                // Hit rate
                if (data.emails_mx_verified > 0) {
                    let hr = Math.round(data.emails_smtp_verified / data.emails_mx_verified * 100);
                    document.getElementById('hitrate').textContent = hr + '%';
                }

                // Countries
                if (data.countries && data.countries.length > 0) {
                    document.getElementById('countries').innerHTML = data.countries.map(c => {
                        let active = c.verified > 0;
                        return `
                        <div style="background:#0a0a0a;border:1px solid ${active?'#00ff41':'#1a3a1a'};
                            border-radius:4px;padding:12px 18px;min-width:130px;text-align:center;cursor:pointer;
                            ${active?'box-shadow:0 0 8px rgba(0,255,65,0.15)':''}"
                            onclick="window.open('/api/country/${c.country}','_blank')">
                            <div style="font-size:16px;font-weight:bold;color:${active?'#00ff41':'#555'};letter-spacing:2px">${c.country}</div>
                            <div style="font-size:28px;color:${active?'#fff':'#333'};margin:6px 0;font-weight:bold;
                                ${active?'text-shadow:0 0 10px #00ff41':''}">
                                ${c.verified}
                            </div>
                            <div style="font-size:10px;color:#555">verified</div>
                            <div style="font-size:10px;color:#333;margin-top:3px">${c.total} total</div>
                        </div>`;
                    }).join('');
                }

                // Domains
                if (data.top_domains) {
                    let maxCnt = Math.max(...data.top_domains.map(d => d.cnt), 1);
                    document.getElementById('domains').innerHTML = data.top_domains.map(d => `
                        <div class="domain-row">
                            <span>${d.domain}</span>
                            <span style="color:#00ff41">${d.smtp_cnt}/${d.cnt}</span>
                        </div>
                        <div class="bar" style="width:${d.cnt/maxCnt*100}%;height:3px;background:#00ff41;border-radius:2px;margin-bottom:5px"></div>
                    `).join('');
                }
            } catch(e) {
                document.getElementById('daemon-status').textContent = 'RECONNECTING...';
            }
        }
        setInterval(pollStats, 2000);
        pollStats();

        // SSE for live email feed
        const evtSource = new EventSource('/api/stream');
        evtSource.onmessage = function(event) {
            const data = JSON.parse(event.data);
            if (data.type === 'email') {
                const feed = document.getElementById('feed');
                let badge = '';
                if (data.data.confidence >= 0.9) badge = '<span class="badge smtp">VERIFIED</span>';
                else if (data.data.confidence >= 0.1 && data.data.confidence < 0.2) badge = '<span class="badge" style="background:#1a1a00;color:#888;border:1px solid #333">CATCH-ALL</span>';
                else if (data.data.mx_valid) badge = '<span class="badge mx">MX</span>';
                else badge = '<span class="badge fail">FAIL</span>';

                const row = document.createElement('div');
                row.className = 'email-row';
                row.innerHTML = `
                    <span class="email">${data.data.email}${badge}</span>
                    <span class="handle">@${data.data.handle}</span>
                `;
                feed.insertBefore(row, feed.firstChild);

                // Keep max 200 rows
                while (feed.children.length > 200) feed.removeChild(feed.lastChild);
            }
        };
        evtSource.onerror = function() {
            document.getElementById('daemon-status').textContent = 'RECONNECTING...';
        };
    </script>
</body>
</html>
"""


def create_app(db_path: str = None) -> Flask:
    """Create Flask dashboard app."""
    app = Flask(__name__)
    app.config["db_path"] = db_path

    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)

    @app.route("/api/stats")
    def api_stats():
        db = EmailDatabase(db_path)
        stats = db.get_stats()
        db.close()
        return jsonify(stats)

    @app.route("/api/stream")
    def api_stream():
        def generate():
            while True:
                try:
                    event = event_queue.get(timeout=5)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    # Send keepalive
                    yield f"data: {json.dumps({'type': 'ping'})}\n\n"

        return Response(
            generate(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/emails")
    def api_emails():
        db = EmailDatabase(db_path)
        emails = db.get_top_emails(limit=100)
        db.close()
        return jsonify(emails)

    @app.route("/api/country/<country>")
    def api_country(country):
        db = EmailDatabase(db_path)
        emails = db.get_emails_by_country(country.upper(), verified_only=True, limit=500)
        db.close()
        if not emails:
            return f"<pre>No verified emails for {country.upper()}</pre>"
        lines = [f"Verified emails for {country.upper()} ({len(emails)}):\n"]
        for e in emails:
            lines.append(f"  {e['email']:<40s} @{e['handle']}")
        return f"<pre style='background:#0a0a0a;color:#00ff41;padding:20px;font-family:monospace'>{'<br>'.join(lines)}</pre>"

    return app
