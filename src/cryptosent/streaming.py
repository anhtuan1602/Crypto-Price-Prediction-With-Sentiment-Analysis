"""Kafka plumbing (report: crawler -> Kafka -> LLM consumer -> MongoDB).

Topics:
  news.raw     article documents published by the crawl DAG
  news.scored  the same documents after sentiment + NLP enrichment
               (consumed by the dashboard API for its live feed)
"""
from __future__ import annotations

import json
import logging
import signal
import time
from typing import Any, Iterable

from .collectors.news_base import Article
from .storage import _json_default

log = logging.getLogger(__name__)


def make_producer(bootstrap: str):
    from kafka import KafkaProducer

    return KafkaProducer(
        bootstrap_servers=bootstrap.split(","),
        value_serializer=lambda v: json.dumps(v, default=_json_default).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        acks="all",
        linger_ms=50,
        retries=5,
    )


def publish_articles(producer, topic: str, articles: Iterable[Article]) -> int:
    n = 0
    for art in articles:
        doc = art.to_doc()
        producer.send(topic, key=doc["_id"], value=doc)
        n += 1
    producer.flush()
    return n


def make_consumer(bootstrap: str, topic: str, group_id: str):
    from kafka import KafkaConsumer

    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap.split(","),
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        value_deserializer=lambda b: json.loads(b.decode("utf-8")),
        max_poll_records=20,
        max_poll_interval_ms=15 * 60 * 1000,  # LLM calls can be slow
    )


def run_sentiment_consumer(settings, store, agent=None) -> None:
    """Long-running service: score every raw article exactly once.

    Offsets are committed only after the MongoDB write, so a crash replays the
    batch instead of losing it; re-scoring is idempotent (same ``_id``)."""
    from .sentiment import build_agent, enrich_documents

    agent = agent or build_agent(settings)
    consumer = make_consumer(settings.kafka_bootstrap, settings.topic_raw, "sentiment-consumer")
    producer = make_producer(settings.kafka_bootstrap)
    running = True

    def stop(*_: Any) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log.info("Sentiment consumer listening on %s (backend=%s)", settings.topic_raw, agent.backend)
    paused_until: float | None = None  # monotonic time; inf = until restart
    while running:
        batch = consumer.poll(timeout_ms=2000)
        docs = [rec.value for records in batch.values() for rec in records]
        if paused_until is not None and time.monotonic() >= paused_until:
            paused_until = None
            if hasattr(agent, "disabled"):
                agent.disabled = None
            log.info("Resuming sentiment scoring")
        if docs:
            store.upsert_articles([Article.from_doc(d) for d in docs])
            if paused_until is None:
                items, halt = enrich_documents(agent, docs)
                if items:
                    store.set_enrichment_many(items)
                    by_id = {d["_id"]: d for d in docs}
                    for article_id, sentiment, nlp in items:
                        producer.send(settings.topic_scored, key=article_id,
                                      value={**by_id[article_id], "sentiment": sentiment, "nlp": nlp})
                if halt is not None:
                    # One clear line. Articles are still stored, unscored; the hourly
                    # sentiment_catchup DAG scores them once the problem is gone.
                    # Quota pauses retry after an hour, config errors need a restart.
                    paused_until = time.monotonic() + 3600 if halt.kind == "quota" else float("inf")
                    log.error("Sentiment scoring paused (%s), articles are stored unscored: %s",
                              "retrying in 1 hour" if halt.kind == "quota" else "fix .env and restart", halt)
            producer.flush()
            consumer.commit()
    consumer.close()
    producer.close()
