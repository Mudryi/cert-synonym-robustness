"""CLI entry point for the CERT-UA research dataset pipeline.

    python main.py crawl --limit 50     # attempt 50 unique articles
    python main.py crawl --all          # full archive
    python main.py crawl --resume       # continue an interrupted crawl
    python main.py parse                # rebuild processed JSONL from the raw cache (offline)
    python main.py chunk                # build text chunks
    python main.py validate             # validation report
    python main.py stats                # corpus statistics
    python main.py pilot                # pilot sample + side-by-side review files
"""

from __future__ import annotations

import argparse
import logging
import sys

from cert_ua import config, storage


def _setup_logging(verbose: bool) -> None:
    config.ensure_dirs()
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stderr),
        logging.FileHandler(config.REPORTS_DIR / "crawl.log", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)


def cmd_crawl(args: argparse.Namespace) -> int:
    from cert_ua.crawler import AccessDenied, Crawler, PoliteClient, check_robots
    from cert_ua.pipeline import build_dataset

    log = logging.getLogger("crawl")
    progress = storage.Progress()
    if args.all and not args.resume:
        # A fresh full crawl re-walks the listing to pick up new articles; cached raw responses are reused.
        progress.listing_exhausted = False
        progress.last_listing_page = None

    with PoliteClient(delay=args.delay) as client:
        robots = check_robots(client)
        log.info("robots.txt: %s", robots.get("policy"))
        if not robots.get("allowed", True):
            log.error("robots.txt disallows crawling the API; aborting")
            return 2
        crawler = Crawler(client, progress)
        try:
            if not progress.listing_exhausted:
                start = (progress.last_listing_page + 1) if (args.resume and progress.last_listing_page is not None) else 0
                # With --limit we only page through the listing until enough IDs are known.
                until = None if args.limit is None else len(crawler.discovered) + args.limit
                crawler.discover_listing(until_count=until, start_page=start)
            stats = crawler.crawl(
                limit=args.limit,
                refresh=args.refresh,
                follow_links=not args.no_follow,
                variants=not args.no_variants,
            )
        except AccessDenied as e:
            log.error("access denied, stopping without retrying: %s", e)
            return 3
        log.info("crawl finished: %s; HTTP requests: %d", stats, client.request_count)

    n = build_dataset()
    log.info("processed dataset: %d records -> %s", n, config.ARTICLES_PATH)
    return 0


def cmd_parse(args: argparse.Namespace) -> int:
    from cert_ua.pipeline import build_dataset

    n = build_dataset()
    print(f"{n} records -> {config.ARTICLES_PATH}")
    return 0


def cmd_chunk(args: argparse.Namespace) -> int:
    from cert_ua.chunker import write_chunks

    n = write_chunks(chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"{n} chunks -> {config.CHUNKS_PATH}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    from cert_ua.validate import run_validation

    summary = run_validation()
    print(f"validation: {summary['passed']}/{summary['total']} records pass "
          f"({summary['pass_rate']:.1%}) -> {config.VALIDATION_REPORT_PATH}")
    return 0 if summary["pass_rate"] >= 0.95 else 1


def cmd_stats(args: argparse.Namespace) -> int:
    from cert_ua.validate import compute_statistics

    stats = compute_statistics()
    storage.write_json(config.STATS_PATH, stats)
    c = stats["counts"]
    print(f"discovered={c['discovered']} parsed={c['parsed']} failed={c['failed']} -> {config.STATS_PATH}")
    return 0


def cmd_pilot(args: argparse.Namespace) -> int:
    from cert_ua.pilot import run_pilot

    ids = run_pilot(n=args.n, fetch=not args.offline, delay=args.delay)
    print(f"pilot: {len(ids)} articles -> {config.REPORTS_DIR / 'pilot'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="CERT-UA research dataset crawler/parser")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("crawl", help="discover and fetch articles, then rebuild the dataset")
    mode = c.add_mutually_exclusive_group(required=True)
    mode.add_argument("--limit", type=int, help="number of unique articles to attempt")
    mode.add_argument("--all", action="store_true", help="crawl the whole archive")
    mode.add_argument("--resume", action="store_true", help="continue from saved progress")
    c.add_argument("--refresh", action="store_true", help="re-download articles even if cached")
    c.add_argument("--no-variants", action="store_true", help="do not fetch uk/en language variants")
    c.add_argument("--no-follow", action="store_true", help="do not follow previous/next links")
    c.add_argument("--delay", type=float, default=config.REQUEST_DELAY_S, help="seconds between requests")
    c.set_defaults(func=cmd_crawl)

    sub.add_parser("parse", help="rebuild processed JSONL from raw cache").set_defaults(func=cmd_parse)

    ch = sub.add_parser("chunk", help="split articles into text chunks")
    ch.add_argument("--chunk-size", type=int, default=config.DEFAULT_CHUNK_SIZE, help="max characters per chunk")
    ch.add_argument("--overlap", type=int, default=config.DEFAULT_CHUNK_OVERLAP, help="overlap in characters")
    ch.set_defaults(func=cmd_chunk)

    sub.add_parser("validate", help="run validation rules").set_defaults(func=cmd_validate)
    sub.add_parser("stats", help="compute corpus statistics").set_defaults(func=cmd_stats)

    pl = sub.add_parser("pilot", help="fetch/parse a pilot sample spread across years and categories")
    pl.add_argument("-n", type=int, default=30)
    pl.add_argument("--offline", action="store_true", help="only use already cached articles")
    pl.add_argument("--delay", type=float, default=config.REQUEST_DELAY_S)
    pl.set_defaults(func=cmd_pilot)

    args = p.parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
