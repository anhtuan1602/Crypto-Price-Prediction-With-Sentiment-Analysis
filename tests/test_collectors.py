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


class FakeJsonSession:
    """Answers get_json from a function of (url, params)."""

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def get_json(self, url, params=None, headers=None, return_errors=False):
        self.calls.append((url, dict(params or {})))
        return self.answer(url, params or {})


def test_parse_current_cryptocompare_items():
    from cryptosent.collectors.cryptocompare import parse_items

    items = [{"TITLE": "BTC news", "URL": "https://c/2", "PUBLISHED_ON": 1766224800, "BODY": "text",
              "SOURCE_DATA": {"NAME": "CoinDesk"}, "CATEGORY_DATA": [{"CATEGORY": "BTC"}]}]
    arts = parse_items(items)
    assert len(arts) == 1 and arts[0].publisher == "CoinDesk" and arts[0].extra["categories"] == "BTC"


def test_cryptocompare_backfill_pages_back_with_data_api():
    from datetime import datetime, timezone

    from cryptosent.collectors.cryptocompare import DATA_API_URL, CryptoCompareCollector

    day = 86_400
    end = 1_766_224_800

    def answer(url, params):
        assert url == DATA_API_URL
        top = params.get("to_ts", end)
        return {"Data": [{"TITLE": f"BTC {t}", "URL": f"https://c/{t}", "PUBLISHED_ON": t}
                         for t in range(top, top - 100 * 3600, -3600)], "Err": {}}

    session = FakeJsonSession(answer)
    arts = CryptoCompareCollector(session=session, api_key="k").backfill(
        datetime.fromtimestamp(end - 10 * day, timezone.utc), datetime.fromtimestamp(end, timezone.utc), pause=0)
    assert len(session.calls) == 3 and 240 <= len(arts) <= 243
    assert min(a.published_at for a in arts).timestamp() >= end - 10 * day
    assert all(c[1]["to_ts"] > end - 11 * day for c in session.calls)


def test_cryptocompare_backfill_falls_back_to_legacy_and_reports_errors():
    from datetime import datetime, timedelta, timezone

    import pytest

    from cryptosent.collectors.cryptocompare import NEWS_URL, CryptoCompareCollector, CryptoCompareError

    end = datetime(2025, 12, 20, tzinfo=timezone.utc)

    def legacy_only(url, params):
        if url != NEWS_URL:
            return {"Data": [], "Err": {"type": 1, "message": "API key is required"}}
        if "lTs" in params and params["lTs"] < end.timestamp() - 3600:
            return {"Response": "Success", "Data": []}
        return {"Response": "Success", "Data": [{"title": "BTC", "url": "https://c/1",
                                                  "published_on": int(end.timestamp()) - 60}]}

    arts = CryptoCompareCollector(session=FakeJsonSession(legacy_only)).backfill(
        end - timedelta(days=1), end, pause=0)
    assert len(arts) == 1

    def broken(url, params):
        return {"Response": "Error", "Message": "You need a valid auth key", "Data": []}

    with pytest.raises(CryptoCompareError, match="CRYPTOCOMPARE_API_KEY.*valid auth key"):
        CryptoCompareCollector(session=FakeJsonSession(broken)).backfill(end - timedelta(days=1), end, pause=0)
    assert CryptoCompareCollector(session=FakeJsonSession(broken)).crawl() == []


class FakeTextResponse:
    def __init__(self, payload=None, text=None):
        import json

        self.text = text if text is not None else json.dumps(payload)
        self._payload = payload

    def json(self):
        import json

        return json.loads(self.text)


class FakeGdeltSession:
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def get(self, url, params=None, headers=None, return_errors=False):
        self.calls.append(dict(params))
        return self.answer(params)


def _gdelt_item(ts, title="Bitcoin rallies"):
    return {"url": f"https://g/{ts:%Y%m%d%H%M%S}/{title}", "title": title, "seendate": f"{ts:%Y%m%dT%H%M%SZ}",
            "domain": "example.com", "language": "English", "sourcecountry": "United States"}


def test_gdelt_parse_filters_irrelevant_titles():
    from datetime import datetime, timezone

    from cryptosent.collectors.gdelt import parse_articles

    t = datetime(2026, 11, 1, 12, tzinfo=timezone.utc)
    arts = parse_articles([_gdelt_item(t), _gdelt_item(t, "Local football results"), {"title": "x"}])
    assert len(arts) == 1 and arts[0].published_at == t and arts[0].source == "gdelt"
    assert arts[0].publisher == "example.com" and arts[0].body == ""


def test_gdelt_backfill_splits_saturated_windows():
    from datetime import datetime, timedelta, timezone

    from cryptosent.collectors.gdelt import MAX_RECORDS, GdeltCollector

    end = datetime(2026, 11, 2, tzinfo=timezone.utc)
    fmt = "%Y%m%d%H%M%S"

    def answer(params):
        lo = datetime.strptime(params["startdatetime"], fmt).replace(tzinfo=timezone.utc)
        hi = datetime.strptime(params["enddatetime"], fmt).replace(tzinfo=timezone.utc)
        if hi - lo > timedelta(hours=3):  # pretend wide windows hit the cap
            return FakeTextResponse({"articles": [_gdelt_item(hi - timedelta(minutes=1))] * MAX_RECORDS})
        if lo < end - timedelta(hours=12):
            return FakeTextResponse(text="{}")  # nothing that far back
        return FakeTextResponse({"articles": [_gdelt_item(lo + timedelta(minutes=m)) for m in range(0, 180, 30)]})

    session = FakeGdeltSession(answer)
    arts = GdeltCollector(session=session, pause=0).backfill(end - timedelta(days=1), end)
    assert len(arts) == 4 * 6  # 12 h of 3-hour halves, 6 articles each
    assert all(a.published_at < end for a in arts)


def test_gdelt_errors_are_loud_on_first_window():
    from datetime import datetime, timedelta, timezone

    import pytest

    from cryptosent.collectors.gdelt import GdeltCollector, GdeltError

    end = datetime(2026, 11, 2, tzinfo=timezone.utc)
    bad = FakeGdeltSession(lambda p: FakeTextResponse(text="Your query was too short or too long."))
    with pytest.raises(GdeltError, match="too short"):
        GdeltCollector(session=bad, pause=0).backfill(end - timedelta(days=1), end)
    assert GdeltCollector(session=bad, pause=0).crawl() == []


def test_backfill_news_uses_gdelt_without_key_and_survives_one_failing_source(tmp_path, monkeypatch):
    from cryptosent.collectors import build_backfill_collectors, build_news_collectors
    from cryptosent.config import Settings
    from cryptosent.pipeline import backfill_news
    from cryptosent.storage import LocalStore

    s = Settings(data_dir=tmp_path, cryptocompare_api_key=None,
                 news_sources=["rss", "cryptocompare"], news_backfill_sources=["gdelt", "cryptocompare"])
    assert [c.name for c in build_backfill_collectors(s)] == ["gdelt"]
    assert [c.name for c in build_news_collectors(s)] == ["rss"]

    s.cryptocompare_api_key = "k"
    assert [c.name for c in build_backfill_collectors(s)] == ["gdelt", "cryptocompare"]

    from datetime import datetime, timezone

    from cryptosent.collectors import news_base
    from cryptosent.collectors.cryptocompare import CryptoCompareCollector, CryptoCompareError
    from cryptosent.collectors.gdelt import GdeltCollector

    art = news_base.Article("https://g/1", "Bitcoin news", datetime(2026, 11, 1, tzinfo=timezone.utc), "gdelt")
    monkeypatch.setattr(GdeltCollector, "backfill", lambda self, start: [art])

    def fail(self, start):
        raise CryptoCompareError("paid only")

    monkeypatch.setattr(CryptoCompareCollector, "backfill", fail)
    assert backfill_news(s, LocalStore(tmp_path), days=3) == 1


def test_fetch_full_text_replaces_teasers_once():
    from datetime import datetime, timezone

    from cryptosent.collectors.news_base import Article, fetch_full_text

    page = "<html><nav>menu</nav><div class='entry-content'>" + "<p>Bitcoin rose sharply today.</p>" * 30 + \
           "</div><footer>footer</footer></html>"

    class Session:
        calls = 0

        def get(self, url, **kw):
            Session.calls += 1
            return type("R", (), {"text": page})()

    t = datetime(2026, 11, 26, tzinfo=timezone.utc)
    teaser = Article("https://n/1", "Bitcoin rises", t, "rss", body="short teaser")
    full = Article("https://n/2", "Bitcoin rises", t, "rss", body="x " * 400)
    assert fetch_full_text(Session(), [teaser, full]) == 1
    assert teaser.body.startswith("Bitcoin rose sharply") and "menu" not in teaser.body and "footer" not in teaser.body
    assert teaser.extra["full_text"] is True and Session.calls == 1
