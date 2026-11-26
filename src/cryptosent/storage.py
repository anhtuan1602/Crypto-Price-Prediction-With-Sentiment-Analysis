"""Persistence layer.

``MongoStore`` is what the Docker stack uses (report: MongoDB 7). ``LocalStore``
keeps the same interface on plain files so the CLI, tests and the synthetic
demo run without any services.

Collections / files:
  articles     one document per article, ``_id`` = SHA-256(url|title),
               enriched in place with ``sentiment`` and ``nlp`` sub-documents
  prices       1-minute candles, ``_id`` = candle open time
  predictions  one document per decision time
  model_runs   evaluation summaries written by training
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

import pandas as pd

from .collectors.news_base import Article, to_utc
from .collectors.prices import PRICE_COLUMNS

log = logging.getLogger(__name__)


class Store(Protocol):
    def upsert_articles(self, articles: Iterable[Article]) -> int: ...
    def articles_missing_sentiment(self, limit: int = 100) -> list[dict[str, Any]]: ...
    def set_enrichment(self, article_id: str, sentiment: dict[str, Any], nlp: dict[str, Any]) -> None: ...
    def get_articles(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame: ...
    def upsert_prices(self, df: pd.DataFrame) -> int: ...
    def get_prices(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame: ...
    def last_price_time(self) -> datetime | None: ...
    def save_predictions(self, rows: list[dict[str, Any]]) -> None: ...
    def get_predictions(self, limit: int = 500) -> list[dict[str, Any]]: ...
    def save_model_run(self, run: dict[str, Any]) -> None: ...
    def latest_model_run(self) -> dict[str, Any] | None: ...


def _json_default(value: Any) -> Any:
    if isinstance(value, (datetime, pd.Timestamp)):
        return to_utc(value).isoformat()
    raise TypeError(f"Not JSON serialisable: {type(value)}")


def articles_frame(docs: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten article documents into one row per article."""
    rows = []
    for doc in docs:
        row = {
            "article_id": doc.get("_id"),
            "url": doc.get("url"),
            "title": doc.get("title"),
            "body": doc.get("body", ""),
            "source": doc.get("source"),
            "publisher": doc.get("publisher"),
            "published_at": to_utc(doc["published_at"]),
        }
        sent = doc.get("sentiment") or {}
        row.update({
            "sentiment_score": sent.get("score"),
            "sentiment_label": sent.get("label"),
            "sentiment_confidence": sent.get("confidence"),
            "sentiment_backend": sent.get("backend"),
        })
        for key, value in (doc.get("nlp") or {}).items():
            row[f"nlp_{key}"] = value
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["article_id", "published_at", "title", "sentiment_score"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df.sort_values("published_at").reset_index(drop=True)


class LocalStore:
    def __init__(self, root: Path | str = "data"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._articles_path = self.root / "articles.json"
        self._prices_path = self.root / "prices.csv"
        self._pred_path = self.root / "predictions.json"
        self._runs_path = self.root / "model_runs.json"

    # -- helpers ---------------------------------------------------------
    def _load(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        return json.loads(path.read_text())

    def _dump(self, path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, default=_json_default))
        tmp.replace(path)

    # -- articles --------------------------------------------------------
    def _articles(self) -> dict[str, dict[str, Any]]:
        return self._load(self._articles_path, {})

    def upsert_articles(self, articles: Iterable[Article]) -> int:
        data = self._articles()
        new = 0
        for art in articles:
            doc = json.loads(json.dumps(art.to_doc(), default=_json_default))
            if doc["_id"] in data:
                continue  # dedup: first copy wins, enrichment is kept
            data[doc["_id"]] = doc
            new += 1
        self._dump(self._articles_path, data)
        return new

    def articles_missing_sentiment(self, limit: int = 100) -> list[dict[str, Any]]:
        docs = [d for d in self._articles().values() if not d.get("sentiment")]
        docs.sort(key=lambda d: d["published_at"])
        return docs[:limit]

    def set_enrichment(self, article_id: str, sentiment: dict[str, Any], nlp: dict[str, Any]) -> None:
        self.set_enrichment_many([(article_id, sentiment, nlp)])

    def set_enrichment_many(self, items: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> None:
        data = self._articles()
        for article_id, sentiment, nlp in items:
            if article_id in data:
                data[article_id]["sentiment"] = sentiment
                data[article_id]["nlp"] = nlp
        self._dump(self._articles_path, data)

    def get_articles(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        df = articles_frame(list(self._articles().values()))
        if df.empty:
            return df
        if start is not None:
            df = df[df["published_at"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["published_at"] < pd.Timestamp(end)]
        return df.reset_index(drop=True)

    # -- prices ----------------------------------------------------------
    def get_prices(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        if not self._prices_path.exists():
            return pd.DataFrame(columns=PRICE_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
        df = pd.read_csv(self._prices_path, index_col="open_time")
        df.index = pd.to_datetime(df.index, utc=True)
        if start is not None:
            df = df[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df[df.index < pd.Timestamp(end)]
        return df

    def upsert_prices(self, df: pd.DataFrame) -> int:
        if df.empty:
            return 0
        existing = self.get_prices()
        before = len(existing)
        merged = pd.concat([existing, df[PRICE_COLUMNS]])
        merged = merged[~merged.index.duplicated(keep="last")].sort_index()
        merged.index.name = "open_time"
        merged.to_csv(self._prices_path)
        return len(merged) - before

    def last_price_time(self) -> datetime | None:
        df = self.get_prices()
        return None if df.empty else df.index.max().to_pydatetime()

    # -- predictions & runs ---------------------------------------------
    def save_predictions(self, rows: list[dict[str, Any]]) -> None:
        data = {r["decision_time"]: r for r in self._load(self._pred_path, [])}
        for row in json.loads(json.dumps(rows, default=_json_default)):
            data[row["decision_time"]] = row
        self._dump(self._pred_path, sorted(data.values(), key=lambda r: r["decision_time"]))

    def get_predictions(self, limit: int = 500) -> list[dict[str, Any]]:
        return self._load(self._pred_path, [])[-limit:]

    def save_model_run(self, run: dict[str, Any]) -> None:
        runs = self._load(self._runs_path, [])
        runs.append(json.loads(json.dumps(run, default=_json_default)))
        self._dump(self._runs_path, runs[-50:])

    def latest_model_run(self) -> dict[str, Any] | None:
        runs = self._load(self._runs_path, [])
        return runs[-1] if runs else None


class MongoStore:
    def __init__(self, uri: str, db: str = "cryptosent"):
        from pymongo import ASCENDING, MongoClient

        self.client = MongoClient(uri, tz_aware=True, serverSelectionTimeoutMS=10_000)
        self.db = self.client[db]
        self.db.articles.create_index([("published_at", ASCENDING)])
        self.db.articles.create_index([("sentiment", ASCENDING)], sparse=True)
        self.db.predictions.create_index([("decision_time", ASCENDING)], unique=True)

    def upsert_articles(self, articles: Iterable[Article]) -> int:
        from pymongo import UpdateOne

        ops = [UpdateOne({"_id": a.article_id}, {"$setOnInsert": a.to_doc()}, upsert=True) for a in articles]
        if not ops:
            return 0
        result = self.db.articles.bulk_write(ops, ordered=False)
        return result.upserted_count

    def articles_missing_sentiment(self, limit: int = 100) -> list[dict[str, Any]]:
        cursor = self.db.articles.find({"sentiment": {"$exists": False}}).sort("published_at", 1).limit(limit)
        return list(cursor)

    def set_enrichment(self, article_id: str, sentiment: dict[str, Any], nlp: dict[str, Any]) -> None:
        self.db.articles.update_one({"_id": article_id}, {"$set": {"sentiment": sentiment, "nlp": nlp}})

    def set_enrichment_many(self, items: list[tuple[str, dict[str, Any], dict[str, Any]]]) -> None:
        for article_id, sentiment, nlp in items:
            self.set_enrichment(article_id, sentiment, nlp)

    def get_articles(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        query: dict[str, Any] = {}
        if start or end:
            query["published_at"] = {}
            if start:
                query["published_at"]["$gte"] = start
            if end:
                query["published_at"]["$lt"] = end
        return articles_frame(list(self.db.articles.find(query)))

    def upsert_prices(self, df: pd.DataFrame) -> int:
        from pymongo import UpdateOne

        if df.empty:
            return 0
        ops = []
        for ts, row in df[PRICE_COLUMNS].iterrows():
            doc = {k: float(v) for k, v in row.items()}
            ops.append(UpdateOne({"_id": ts.to_pydatetime()}, {"$set": doc}, upsert=True))
        result = self.db.prices.bulk_write(ops, ordered=False)
        return result.upserted_count

    def get_prices(self, start: datetime | None = None, end: datetime | None = None) -> pd.DataFrame:
        query: dict[str, Any] = {}
        if start or end:
            query["_id"] = {}
            if start:
                query["_id"]["$gte"] = start
            if end:
                query["_id"]["$lt"] = end
        docs = list(self.db.prices.find(query).sort("_id", 1))
        if not docs:
            return pd.DataFrame(columns=PRICE_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="open_time"))
        df = pd.DataFrame(docs).rename(columns={"_id": "open_time"})
        df["open_time"] = pd.to_datetime(df["open_time"], utc=True)
        return df.set_index("open_time")[PRICE_COLUMNS]

    def last_price_time(self) -> datetime | None:
        doc = self.db.prices.find_one(sort=[("_id", -1)])
        return to_utc(doc["_id"]) if doc else None

    def save_predictions(self, rows: list[dict[str, Any]]) -> None:
        from pymongo import ReplaceOne

        if rows:
            self.db.predictions.bulk_write(
                [ReplaceOne({"decision_time": r["decision_time"]}, r, upsert=True) for r in rows]
            )

    def get_predictions(self, limit: int = 500) -> list[dict[str, Any]]:
        docs = list(self.db.predictions.find({}, {"_id": 0}).sort("decision_time", -1).limit(limit))
        return docs[::-1]

    def save_model_run(self, run: dict[str, Any]) -> None:
        self.db.model_runs.insert_one({**run, "created_at": datetime.now(timezone.utc)})

    def latest_model_run(self) -> dict[str, Any] | None:
        return self.db.model_runs.find_one({}, {"_id": 0}, sort=[("created_at", -1)])


def get_store(settings) -> Store:
    if settings.mongo_uri:
        return MongoStore(settings.mongo_uri, settings.mongo_db)
    log.info("MONGO_URI not set, using LocalStore at %s", settings.data_dir)
    return LocalStore(settings.data_dir)
