"""
CLI interface for X Pathfinder.

Commands:
    run                - Start background daemon (continuous discovery)
    emails <niche>     - One-shot email discovery
    status             - Show DB stats
    export             - Export verified emails to CSV
    discover <niche>   - Account discovery only
"""

import argparse
import asyncio
import sys
import logging

from .database import EmailDatabase


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


async def cmd_run(args):
    """Start daemon + dashboard."""
    from .daemon import EmailDaemon
    from .dashboard import create_app
    import threading

    countries = args.countries.upper().split(",") if args.countries else None

    print(f"\n=== X Pathfinder ===")
    print(f"Workers: {args.workers}")
    print(f"SMTP+Catch-All: always on")
    print(f"Countries: {countries or 'ALL'}")
    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Only verified emails are stored.")
    print(f"Press Ctrl+C to stop\n")

    # Start Flask dashboard in background thread
    app = create_app()
    flask_thread = threading.Thread(
        target=lambda: app.run(
            host="0.0.0.0",
            port=args.port,
            debug=False,
            use_reloader=False,
        ),
        daemon=True,
    )
    flask_thread.start()

    # Run daemon in main thread
    daemon = EmailDaemon(
        workers=args.workers,
        countries=countries,
    )
    await daemon.run()


async def cmd_emails(args):
    """One-shot email discovery."""
    from .email_discoverer import EmailDiscoverer

    print(f"\n=== Email Discovery ({args.niche}) ===\n")

    def progress(gen, stats):
        crisis = " [CRISIS]" if stats.get("crisis_mode") else ""
        print(
            f"  Gen {gen}: +{stats.get('new_verified', 0)} verified, "
            f"total={stats.get('total_verified', 0)}, "
            f"fitness={stats.get('best_fitness', 0):.1f}{crisis}"
        )

    discoverer = EmailDiscoverer(
        niche=args.niche,
        max_concurrent=args.concurrent,
        verify_smtp=not args.no_smtp,
    )

    results = await discoverer.run(
        generations=args.generations,
        on_progress=progress,
    )

    summary = discoverer.get_summary()
    print(f"\n=== Results ===")
    print(f"Accounts targeted: {summary['accounts_targeted']}")
    print(f"Emails total: {summary['total_emails']}")
    print(f"MX verified: {summary['mx_verified']}")
    print(f"SMTP verified: {summary['smtp_verified']}")
    print()

    if results:
        for i, e in enumerate(results[:args.top], 1):
            mx = "MX" if e.mx_valid else "  "
            smtp = "SMTP" if e.smtp_valid is True else (
                "????" if e.smtp_valid is None else "    "
            )
            print(
                f"  {i:2d}. {e.email:<40s} [{mx}|{smtp}] "
                f"conf={e.confidence:.1f} @{e.handle}"
            )


def cmd_status(args):
    """Show database stats."""
    db = EmailDatabase()
    stats = db.get_stats()

    print(f"\n=== X Pathfinder Database ===\n")
    print(f"Accounts:       {stats['accounts']:,}")
    print(f"Emails total:   {stats['emails_total']:,}")
    print(f"MX verified:    {stats['emails_mx_verified']:,}")
    print(f"SMTP verified:  {stats['emails_smtp_verified']:,}")
    print(f"Runs:           {stats['runs']}")

    if stats["top_domains"]:
        print(f"\nTop domains:")
        for d in stats["top_domains"]:
            print(
                f"  {d['domain']:<30s} "
                f"total={d['cnt']}, mx={d['mx_cnt']}, smtp={d['smtp_cnt']}"
            )

    db.close()


def cmd_export(args):
    """Export emails to CSV."""
    db = EmailDatabase()

    verified_only = not args.all
    filepath = args.output or "emails_export.csv"
    country = getattr(args, "country", None)

    db.export_csv(filepath, verified_only=verified_only, country=country)
    count = db.get_email_count(verified_only=verified_only)
    db.close()

    label = "verified" if verified_only else "all"
    extra = f" ({country.upper()})" if country else ""
    print(f"Exported {label} emails{extra} to {filepath}")


async def cmd_discover(args):
    """Account discovery."""
    from .account_discoverer import AccountDiscoverer

    print(f"\n=== Account Discovery ({args.niche}) ===\n")

    def progress(gen, stats):
        print(
            f"  Gen {gen}: +{stats.get('new_accounts', 0)} new, "
            f"fitness={stats.get('best_fitness', 0):.1f}"
        )

    discoverer = AccountDiscoverer(
        niche=args.niche,
        max_concurrent=args.concurrent,
    )

    results = await discoverer.run(
        generations=args.generations,
        on_progress=progress,
    )

    print(f"\nDiscovered: {len(results)} accounts")
    for i, a in enumerate(results[:args.top], 1):
        print(f"  {i}. @{a.handle} fitness={a.fitness_score:.1f}")


async def cmd_backers(args):
    """Discover + score potential crowdfunding backers."""
    from .account_discoverer import AccountDiscoverer
    from .backer_score import BackerScoreEvaluator

    print(f"\n=== Backer Discovery ({args.niche}) ===")
    print(f"Min score: {args.min_score} | Target: {args.target} hot leads\n")

    # Phase 1: Discover accounts
    def progress(gen, stats):
        print(
            f"  Gen {gen}: +{stats.get('new_accounts', 0)} new, "
            f"fitness={stats.get('best_fitness', 0):.1f}"
        )

    discoverer = AccountDiscoverer(
        niche=args.niche,
        max_concurrent=args.concurrent,
    )

    accounts = await discoverer.run(
        generations=args.generations,
        on_progress=progress,
    )

    if not accounts:
        print("No accounts found.")
        return

    # Phase 2: Score for backer potential
    print(f"\nScoring {len(accounts)} accounts for backer potential...\n")

    custom_kw = args.keywords.split(",") if args.keywords else None
    evaluator = BackerScoreEvaluator(custom_keywords=custom_kw)
    results = evaluator.batch_evaluate(accounts)
    stats = evaluator.summary(results)

    # Phase 3: Report
    hot = [r for r in results if r.total >= args.min_score]

    print(f"{'='*60}")
    print(f"  BACKER SCORE RESULTS")
    print(f"{'='*60}")
    print(f"  Total accounts:  {stats['total']}")
    print(f"  Hot  (>=60):     {stats['hot']}")
    print(f"  Warm (35-59):    {stats['warm']}")
    print(f"  Cold (<35):      {stats['cold']}")
    print(f"  Avg score:       {stats['avg_score']}")
    print(f"{'='*60}\n")

    # Show top results
    show_count = min(args.top, len(results))
    for i, r in enumerate(results[:show_count], 1):
        tier_icon = {"hot": "🔥", "warm": "🟡", "cold": "⚪"}.get(r.tier, " ")
        reasons_str = " | ".join(r.reasons[:3]) if r.reasons else ""
        print(
            f"  {i:3d}. {tier_icon} @{r.handle:<25s} "
            f"score={r.total:5.1f} ({r.tier:4s}) "
            f"[oss={r.open_source:.0f} ai={r.ai_interest:.0f} "
            f"bld={r.builder:.0f} eng={r.engagement:.0f} "
            f"reach={r.reach:.0f}]"
        )
        if reasons_str:
            print(f"       {reasons_str}")

    # Export hot leads
    if args.output and hot:
        import json
        export = [r.to_dict() for r in hot]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(export, f, indent=2, ensure_ascii=False)
        print(f"\n  Exported {len(hot)} hot leads to {args.output}")

    print(f"\n  Next step: feed hot leads into pipeline_orchestrator.py")


def main():
    parser = argparse.ArgumentParser(
        prog="x-pathfinder",
        description="Evolutionary X/Twitter email discovery",
    )
    parser.add_argument("-v", "--verbose", action="store_true")

    sub = parser.add_subparsers(dest="command")

    # run (daemon)
    run_p = sub.add_parser("run", help="Start background daemon")
    run_p.add_argument(
        "--workers", "-w", type=int, default=20,
        help="Concurrent workers [default: 20]",
    )
    run_p.add_argument(
        "--port", "-p", type=int, default=8420,
        help="Dashboard port [default: 8420]",
    )
    run_p.add_argument(
        "--countries", type=str, default=None,
        help="Countries to target (comma-separated: US,DE,GB,FR,CH,RU,IN)",
    )

    # emails (one-shot)
    email_p = sub.add_parser("emails", help="One-shot email discovery")
    email_p.add_argument("niche", nargs="?", default="ai")
    email_p.add_argument("--generations", "-g", type=int, default=10)
    email_p.add_argument("--concurrent", "-c", type=int, default=5)
    email_p.add_argument("--top", type=int, default=30)
    email_p.add_argument("--no-smtp", action="store_true")

    # status
    sub.add_parser("status", help="Show DB stats")

    # export
    export_p = sub.add_parser("export", help="Export emails to CSV")
    export_p.add_argument("-o", "--output", default="emails_export.csv")
    export_p.add_argument("--all", action="store_true", help="Include unverified")
    export_p.add_argument("--country", type=str, default=None, help="Filter by country (US, DE, GB, ...)")

    # discover
    disc_p = sub.add_parser("discover", help="Account discovery only")
    disc_p.add_argument("niche", nargs="?", default="ai")
    disc_p.add_argument("--generations", "-g", type=int, default=15)
    disc_p.add_argument("--concurrent", "-c", type=int, default=3)
    disc_p.add_argument("--top", type=int, default=20)

    # backers (discover + score for crowdfunding)
    back_p = sub.add_parser("backers", help="Find & score potential crowdfunding backers")
    back_p.add_argument("niche", nargs="?", default="ai")
    back_p.add_argument("--generations", "-g", type=int, default=15)
    back_p.add_argument("--concurrent", "-c", type=int, default=3)
    back_p.add_argument("--top", type=int, default=50)
    back_p.add_argument("--min-score", type=float, default=60.0, help="Min backer score for hot leads [default: 60]")
    back_p.add_argument("--target", type=int, default=200, help="Target number of hot leads")
    back_p.add_argument("--keywords", type=str, default=None, help="Extra keywords (comma-separated: vibemind,ai agent,mcp)")
    back_p.add_argument("-o", "--output", default="hot_backers.json", help="Export hot leads JSON")

    args = parser.parse_args()
    setup_logging(args.verbose)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    if args.command == "run":
        asyncio.run(cmd_run(args))
    elif args.command == "emails":
        asyncio.run(cmd_emails(args))
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "export":
        cmd_export(args)
    elif args.command == "discover":
        asyncio.run(cmd_discover(args))
    elif args.command == "backers":
        asyncio.run(cmd_backers(args))


if __name__ == "__main__":
    main()
