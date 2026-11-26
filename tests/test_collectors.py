import json
from datetime import datetime, timezone

import pandas as pd

from cryptosent.collectors.coinmarketcap import parse_article_body, parse_content_item, parse_next_data
from cryptosent.collectors.cryptocompare import parse_items
from cryptosent.collectors.news_base import Article, article_hash, clean_text, dedupe, to_utc
from cryptosent.collectors.prices import klines_to_frame
from cryptosent.collectors.rss import parse_feed


def test_to_utc_handles_formats():
    expected = datetime(2025, 12, 13, 14, 19, tzinfo=timezone.utc)
    assert to_utc("2025-12-13T14:19:00Z") == expected
    assert to_utc("2025-12-13T21:19:00+07:00") == expected
    assert to_utc("Sat, 13 Dec 2025 14:19:00 GMT") == expected
    assert to_utc(int(expected.timestamp())) == expected
    assert to_utc(int(expected.timestamp() * 1000)) == expected


def test_clean_text_strips_html_and_noise():
    raw = "<div><script>var x=1</script><p>Bitcoin   rallies&nbsp;5%!</p><style>p{}</style></div>"
    assert clean_text(raw) == "Bitcoin rallies 5%!"


def test_hash_ignores_tracking_params_and_case():
    a = article_hash("https://x.com/news/1?utm_source=tw", "Bitcoin up")
    b = article_hash("https://X.com/news/1/", "  bitcoin UP ")
    assert a == b


def test_dedupe_keeps_first():
    ts = datetime(2025, 1, 1, tzinfo=timezone.utc)
    arts = [Article("https://a/1", "T", ts, "x"), Article("https://a/1?ref=2", "T", ts, "y")]
    assert [a.source for a in dedupe(arts)] == ["x"]


def test_parse_cmc_content_item_news_and_alexandria():
    news = {"slug": "abc", "meta": {"title": "BTC ETF inflows", "subtitle": "Big day", "sourceName": "CoinDesk",
                                    "sourceUrl": "https://coindesk.com/a", "releasedAt": "2025-12-20T10:00:00.000Z",
                                    "type": "news"}}
    art = parse_content_item(news)
    assert art.url == "https://coindesk.com/a" and art.publisher == "CoinDesk"
    assert art.published_at == datetime(2025, 12, 20, 10, tzinfo=timezone.utc)
    alex = {"slug": "what-is-btc", "meta": {"title": "What is BTC", "createdAt": "2025-12-20T10:00:00Z",
                                           "type": "alexandria"}}
    assert parse_content_item(alex).url.endswith("/academy/article/what-is-btc")
    assert parse_content_item({"meta": {"title": ""}}) is None


def test_parse_next_data_walks_nested_payload():
    payload = {"props": {"pageProps": {"news": [
        {"slug": "s1", "meta": {"title": "One", "releasedAt": "2025-12-20T10:00:00Z", "sourceUrl": "https://a/1"}},
        {"slug": "s2", "meta": {"title": "Two", "releasedAt": "2025-12-20T11:00:00Z", "sourceUrl": "https://a/2"}},
    ]}}}
    html = f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(payload)}</script></html>'
    assert [a.title for a in parse_next_data(html)] == ["One", "Two"]


def test_parse_article_body_prefers_article_content():
    body = "word " * 100
    html = f'<html><nav>menu</nav><div class="article-content"><p>{body}</p></div></html>'
    assert parse_article_body(html).startswith("word word")


def test_parse_rss_filters_btc():
    xml = """<?xml version="1.0"?><rss version="2.0"><channel><title>t</title>
      <item><title>Bitcoin hits record</title><link>https://n/1</link>
        <pubDate>Sat, 20 Dec 2025 10:00:00 GMT</pubDate><description>&lt;p&gt;BTC up&lt;/p&gt;</description></item>
      <item><title>Celebrity gossip</title><link>https://n/2</link>
        <pubDate>Sat, 20 Dec 2025 10:00:00 GMT</pubDate><description>nothing</description></item>
    </channel></rss>"""
    arts = parse_feed(xml, "coindesk")
    assert len(arts) == 1 and arts[0].body == "BTC up" and arts[0].publisher == "coindesk"


def test_parse_cryptocompare_items():
    items = [{"title": "BTC news", "url": "https://c/1", "published_on": 1766224800, "body": "text",
              "source_info": {"name": "Decrypt"}}, {"title": "", "url": "x", "published_on": 1}]
    arts = parse_items(items)
    assert len(arts) == 1 and arts[0].publisher == "Decrypt"


def test_klines_to_frame():
    rows = [[1766224800000, "100", "101", "99", "100.5", "2", 1766224859999, "201", 10, "1", "100", "0"],
            [1766224860000, "100.5", "102", "100", "101", "3", 1766224919999, "303", 12, "1", "100", "0"]]
    df = klines_to_frame(rows)
    assert list(df["close"]) == [100.5, 101.0]
    assert df.index[0] == pd.Timestamp("2025-12-20T10:00:00Z")
