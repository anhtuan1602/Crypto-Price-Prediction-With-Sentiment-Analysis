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
    from .sentiment import build_agent, enrich_document

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
    while running:
        batch = consumer.poll(timeout_ms=2000)
        for records in batch.values():
            for rec in records:
                doc = rec.value
                store.upsert_articles([Article.from_doc(doc)])
                sentiment, nlp = enrich_document(agent, doc)
                store.set_enrichment(doc["_id"], sentiment, nlp)
                producer.send(settings.topic_scored, key=doc["_id"], value={**doc, "sentiment": sentiment, "nlp": nlp})
        if batch:
            producer.flush()
            consumer.commit()
    consumer.close()
    producer.close()
