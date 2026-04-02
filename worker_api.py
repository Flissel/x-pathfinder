"""
Remote verification worker.

Deploy this on cheap VPS instances. Each worker runs a simple
Flask API that does SMTP verification from its own IP.

Usage on VPS:
    pip install flask
    python worker_api.py --port 5000 --key YOUR_SECRET

The daemon sends verification requests to multiple workers,
rotating between them. If one IP gets blocked, others continue.
"""

import smtplib
import random
import string
import logging
from flask import Flask, request, jsonify

app = Flask(__name__)
API_KEY = "changeme"  # Set via --key or env WORKER_KEY

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache MX lookups and catch-all results
mx_cache = {}
catchall_cache = {}


def get_mx(domain):
    if domain in mx_cache:
        return mx_cache[domain]
    try:
        import dns.resolver
        answers = dns.resolver.resolve(domain, "MX")
        mx = str(answers[0].exchange).rstrip(".")
        mx_cache[domain] = mx
        return mx
    except Exception:
        mx_cache[domain] = domain
        return domain


def smtp_check(email):
    """Returns: (code, is_catchall, error)"""
    domain = email.split("@")[1]
    mx = get_mx(domain)

    try:
        s = smtplib.SMTP(timeout=10)
        s.connect(mx, 25)
        s.ehlo("verify.local")
        s.mail("verify@verify.local")
        code, msg = s.rcpt(email)
        s.quit()
        return code, False, None
    except Exception as e:
        return -1, False, str(e)[:100]


def check_catchall(domain):
    if domain in catchall_cache:
        return catchall_cache[domain]

    rand = ''.join(random.choices(string.ascii_lowercase + string.digits, k=20))
    fake = f"xpf_catchall_{rand}@{domain}"
    code, _, _ = smtp_check(fake)
    is_catchall = code == 250
    catchall_cache[domain] = is_catchall
    return is_catchall


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/verify", methods=["POST"])
def verify():
    """Verify a single email."""
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    data = request.json
    email = data.get("email", "")
    if not email or "@" not in email:
        return jsonify({"error": "invalid email"}), 400

    domain = email.split("@")[1]

    # Check catch-all first
    if check_catchall(domain):
        return jsonify({
            "email": email,
            "exists": None,
            "catch_all": True,
            "code": -1,
        })

    code, _, error = smtp_check(email)

    return jsonify({
        "email": email,
        "exists": code == 250,
        "catch_all": False,
        "code": code,
        "error": error,
    })


@app.route("/verify_batch", methods=["POST"])
def verify_batch():
    """Verify a batch of emails."""
    if request.headers.get("X-API-Key") != API_KEY:
        return jsonify({"error": "unauthorized"}), 401

    data = request.json
    emails = data.get("emails", [])
    results = []

    # Group by domain for efficiency
    by_domain = {}
    for email in emails:
        domain = email.split("@")[1]
        by_domain.setdefault(domain, []).append(email)

    for domain, domain_emails in by_domain.items():
        # Check catch-all once per domain
        if check_catchall(domain):
            for email in domain_emails:
                results.append({
                    "email": email,
                    "exists": None,
                    "catch_all": True,
                    "code": -1,
                })
            continue

        # Verify each email
        for email in domain_emails:
            code, _, error = smtp_check(email)
            results.append({
                "email": email,
                "exists": code == 250,
                "catch_all": False,
                "code": code,
                "error": error,
            })

    return jsonify({"results": results})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--key", type=str, default="changeme")
    args = parser.parse_args()

    API_KEY = args.key
    app.run(host="0.0.0.0", port=args.port)
