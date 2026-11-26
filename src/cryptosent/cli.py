"""Command line entry point: ``cryptosent <command>`` (or ``python -m cryptosent``)."""
from __future__ import annotations

import argparse
import json
import logging
import sys

from .config import get_settings
from .storage import get_store


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cryptosent", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("crawl", help="run the live news collectors once")
    p = sub.add_parser("backfill-news", help="load historical news (GDELT, plus CryptoCompare with a key)")
    p.add_argument("--days", type=int, default=90)
    p = sub.add_parser("prices", help="sync Binance 1m candles")
    p.add_argument("--backfill-days", type=int, default=None)
    p = sub.add_parser("enrich", help="score articles that have no sentiment yet")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--backend", default=None, help="override SENTIMENT_BACKEND for this run (e.g. lexicon)")
    p = sub.add_parser("rescore", help="re-score articles scored earlier by another backend")
    p.add_argument("--from-backend", default="lexicon")
    p.add_argument("--limit", type=int, default=500)
    p = sub.add_parser("dataset", help="build the modelling table and write it to CSV")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--out", default="artifacts/dataset.csv")
    p = sub.add_parser("train", help="run the full evaluation and train the live model")
    p.add_argument("--days", type=int, default=None)
    p.add_argument("--models", nargs="*", default=None)
    sub.add_parser("predict", help="predict recent decision times")
    sub.add_parser("consume", help="run the Kafka sentiment consumer (long-running)")
    p = sub.add_parser("demo", help="offline end-to-end run on synthetic data")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--models", nargs="*", default=None)
    p = sub.add_parser("serve", help="start the API and dashboard")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)

    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    settings = get_settings()

    if args.command == "serve":
        import uvicorn

        uvicorn.run("cryptosent.api.app:app", host=args.host, port=args.port)
        return 0

    from . import pipeline

    store = get_store(settings)
    if args.command == "crawl":
        result = pipeline.crawl_news(settings, store)
    elif args.command == "backfill-news":
        result = {"new_articles": pipeline.backfill_news(settings, store, args.days)}
    elif args.command == "prices":
        result = {"new_candles": pipeline.fetch_prices(settings, store, args.backfill_days)}
    elif args.command == "enrich":
        result = {"scored": pipeline.enrich_pending(settings, store, args.limit, backend=args.backend)}
    elif args.command == "rescore":
        result = {"rescored": pipeline.rescore(settings, store, args.from_backend, args.limit)}
    elif args.command == "dataset":
        ds = pipeline.load_dataset(settings, store, args.days)
        ds.frame.to_csv(args.out)
        result = {"rows": len(ds.frame), "path": args.out}
    elif args.command == "train":
        report = pipeline.train(settings, store, args.days, args.models)
        result = {"best_model": report["best_model"], "test": report["test"], "backtest": report["backtest"]["cost_aware"]}
    elif args.command == "predict":
        result = {"predictions": len(pipeline.predict_latest(settings, store))}
    elif args.command == "consume":
        from .streaming import run_sentiment_consumer

        run_sentiment_consumer(settings, store)
        return 0
    elif args.command == "demo":
        report = pipeline.run_demo(settings, store, args.days, args.models)
        result = {"best_model": report["best_model"], "test": report["test"], "backtest": report["backtest"]["cost_aware"]}
    else:  # pragma: no cover
        parser.error(f"unknown command {args.command}")
        return 2
    json.dump(pipeline.json_safe(result), sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
