# Crypto Price Prediction With Sentiment Analysis

An end-to-end web-mining system that crawls crypto news, scores each article
with an LLM (Gemini 2.5 Flash), aggregates sentiment into time windows and
predicts short-term Bitcoin price moves. It follows the design in the Group 08
technical report and fixes the issues that would make its results hard to
defend (see [What changed from the report](#what-changed-from-the-report)).

```
 CoinMarketCap ─┐                                    ┌─> MongoDB (articles, sentiment, nlp)
 RSS feeds ─────┼─ Airflow cmc_news_10m ─> Kafka ────┤
 GDELT history ─┘   (crawl, clean, dedup)  news.raw   └─ sentiment-consumer (Gemini + NLP) ─> news.scored
 Binance 1m ────── Airflow btc_prices_5m ─────────────> MongoDB (prices)
                   Airflow train_daily ─> leakage-safe dataset ─> walk-forward eval ─> MLflow + model
                   Airflow predict_10m ─> predictions ─> FastAPI ─> dashboard
```

## Quick start (offline demo, no services, about 1 minute)

```bash
pip install -e ".[dev]"
cryptosent demo --days 30      # synthetic prices + news, full train/eval
cryptosent serve               # dashboard on http://localhost:8000
pytest                         # 76 tests
```

The demo plants a small, delayed news effect in a random-walk price series.
It checks the whole pipeline: news features should beat price-only
features, and the shuffled-target control should score about zero. **Demo
numbers are not real results.**

## Full stack (Docker)

```bash
cp .env.example .env           # add GEMINI_API_KEY, set an Airflow password
docker compose up -d --build
```

| Service | URL |
|---|---|
| Dashboard / API | http://localhost:8000 (API docs at `/docs`) |
| Airflow | http://localhost:8080 (user and password from `.env`) |
| MLflow | http://localhost:5000 |

Then, once, trigger the **`backfill_history`** DAG in Airflow. By default it
loads 90 days of 1-minute Binance candles only. News comes from live
collection, as in the original report: from the moment the stack is up,
`cmc_news_10m` crawls full articles every 10 minutes and Gemini scores them.
`train_daily` trains only on the period since the first live crawl
(`TRAIN_START=auto`). It skips with a log line until there are 200 samples
(about 33 hours). Five days give about 700 samples.

The scheduled DAGs keep everything current:

| DAG | Schedule | What it does |
|---|---|---|
| `cmc_news_10m` | every 10 min | crawl CoinMarketCap + RSS, dedupe, store, publish to Kafka |
| `btc_prices_5m` | every 5 min | sync Binance 1m candles |
| `predict_10m` | every 10 min | predict the next window with the latest model |
| `sentiment_catchup_1h` | hourly | score anything the Kafka consumer missed, upgrade some lexicon scores to Gemini |
| `train_daily` | 02:00 UTC | score leftovers, retrain on the live-news period, log to MLflow (also trigger it by hand) |
| `backfill_history` | manual | price history; historical news only if `news_days` > 0 |

### Live news sources (no keys)

* **CoinMarketCap**: the site's news content endpoint, with an HTML fallback.
* **RSS** from 30 crypto outlets (see `DEFAULT_FEEDS` in
  `collectors/rss.py`). Add your own with
  `RSS_FEEDS_EXTRA=name=https://site/feed,...` and switch feeds off with
  `RSS_FEEDS_DISABLED=name,...`.
* `cryptosent sources` fetches every source once and prints a table: HTTP
  status, feed entries, articles kept and the newest article. It shows any
  feed that is blocked or has moved.
* `cryptosent stats --hours 24` shows how many articles were collected per
  publisher and per day, and how many were scored by each backend.
* When a feed has only a teaser, the article page is downloaded **once** for
  each new article and the main text is extracted (`FETCH_FULL_TEXT=true`,
  at most 60 pages per run). A broken feed or a blocked page is skipped.
* Volume follows US and EU working hours. Overnight in UTC (roughly 22:00
  to 06:00) adds only a trickle, so judge volume over a full day with
  `cryptosent stats`.

### Historical news (optional backfill)

Off by default. Set `news_days` on `backfill_history` to use it. It mixes
headline-only articles into training, so the default keeps to live full text.

* **GDELT** (free, no key) searches news from about the **last 3
  months** only. It returns headlines and links but no article text, so
  those articles are scored on the title. Its timestamp is when GDELT first
  saw the article, usually 15 to 60 minutes after publication. That delay
  can only make news late, never early, so it cannot leak future data. GDELT
  allows about one request every 5 seconds, so a 90-day backfill takes
  roughly an hour. Titles are filtered for crypto terms.
* **CryptoCompare** (CoinDesk Data) has article text and a longer history,
  but its free tier ended in May 2026. It is used only when
  `CRYPTOCOMPARE_API_KEY` is set.
* For more than 3 months of history you need a paid source or GDELT's raw
  GKG files (large downloads, not supported here). Over 3 months the model
  still sees thousands of headlines, which is enough to compare news against
  price-only features.
* If one source fails, the others still run, and the log names the source
  and its error.

### Gemini on the free tier

* The client scores `GEMINI_BATCH_SIZE` articles (default 10) per request
  and paces calls to `GEMINI_RPM` (default 10). It stores each result, so no
  article is scored twice.
* If a model answers 404, the client switches to `gemini-flash-latest`, then
  to other Flash models your key lists, trying at most 3 models.
* After a 400 the client drops optional request fields one at a time
  (thinkingConfig, topP/topK, schema bounds, responseSchema, maxOutputTokens)
  until the model accepts the request. If the error names a field, it drops
  that field straight away. It logs which fields it dropped and keeps that
  setup, so this costs a few calls once per process. A rejected key, an
  unknown model or a request that still fails gives **one** clear error, not
  a retry storm.
* A 503 ("high demand") is retried after 10, 20 and 30 seconds. If it still
  fails, that batch is scored with the lexicon, so no article is lost. The
  hourly `sentiment_catchup_1h` job then upgrades it to Gemini. In `auto` mode the rest of the run then uses the offline lexicon
  (upgrade later with `cryptosent rescore`).
* When the daily quota runs out, scoring stops and the remaining articles
  wait for the next run. The API key goes in a request header and is
  redacted from logs.
* **Live volume vs quota.** New articles arrive a few at a time every 10
  minutes, so the scorer sends at most about one request per crawl: 150 to
  250 requests a day for 300 to 600 articles, plus 5 an hour for upgrading
  lexicon scores. Check your model's requests-per-day limit in Google AI
  Studio (Settings, Rate limits). Flash-Lite free tiers have allowed far more
  than this. If the quota runs out, scoring pauses and catches up later.
* A 90-day historical backfill is tens of thousands of articles, too many for a free
  daily quota. Trigger `backfill_history` with `sentiment_backend=lexicon`.
  The hourly `sentiment_catchup_1h` job then upgrades
  `GEMINI_RESCORE_PER_HOUR` of those articles to Gemini, newest first.
  To do it by hand, run `cryptosent rescore --from-backend lexicon --limit N`.

## How a sample is built

For each decision time *t* on a 10-minute grid (all windows configurable in `.env`):

```
 [t-30m ........ t)   [t ...... t+10m)   [t+10m ...... t+20m]
  observation          lag (unused)        prediction window
  news + candles                           target = % change
```

* News features use only articles with `published_at < t`.
* Price features use only candles that closed by *t*.
* `audit_dataset` checks every row for look-ahead and the tests in
  `tests/test_dataset.py` enforce it.

**Features (65):**
* **Technical (21):** returns, volatility, MA/Bollinger distances, RSI, MACD and volume.
* **LLM sentiment (16):** article count; mean, std, min and max score;
  confidence-weighted S_market; label shares; manipulation count; a
  recency-weighted score; news velocity; 6-hour context.
* **NLP aggregates (15):** polarity ratios, text statistics, BTC/price/technical
  keyword counts, hype phrases and named entities.
* **Time of day / week (4).**

## Evaluation protocol

1. The last 15% of time is the untouched test period.
2. Model selection uses expanding-window walk-forward CV on the rest. An
   embargo of obs+lag+pred minutes separates train and test, so overlapping
   windows cannot leak.
3. Ablation compares `technical_only`, `news_only`, `technical_plus_news` and
   `all_with_time`, using the same model and split.
4. Baselines: no-change, train mean, AR(1) on the last return, and momentum.
5. Sanity checks: target autocorrelation, a permutation test, train R² (to
   spot memorisation) and the timestamp audit.
6. Metrics are MAE, RMSE, R², **direction accuracy** and the information
   coefficient. A **fee-aware backtest** (0.1% per side, non-overlapping
   trades) checks whether the predictions are economically useful.

Results go to `artifacts/report.md`, `report.json`, `test_predictions.csv`,
`model.joblib` and MLflow.

## What changed from the report

| Report | Here | Why |
|---|---|---|
| CoinGecko prices | Binance 1m klines | CoinGecko's free API has no 1-minute history |
| CoinMarketCap only, 20 days | + RSS + GDELT backfill (CryptoCompare with a key) | months of history and redundancy |
| Prompt returns label + confidence | label + **score in [-1, 1]** + confidence + reasoning (JSON schema) | the S_market and std/min/max features need a score |
| Lag described but audit only checked `article < window end` | lag enforced in code, audited on every row, unit tested | prevents same-window correlation counted as prediction |
| Scaler / noise / correlation filter | fitted inside each training fold | no test statistics leak into training |
| Single 70/15/15 split | walk-forward CV + embargo + final holdout | robust to one lucky period |
| MA / Bollinger as dollar levels | relative to price | stationary, no regime memorisation |
| R² only | + direction accuracy, IC, backtest with fees | R² alone hides whether it is tradable |
| XGBoost in figures only | LightGBM in the model zoo | fast, regularised boosting |

**Expected results on real data.** 10-minute BTC returns are close to
unpredictable. A leak-free model will likely score R² near 0 to 0.05 and
direction accuracy around 51 to 56%. A much higher R² should be treated as a
bug until proven otherwise. Try `WINDOW_PRED_MIN=60` too, because news effects
are usually easier to detect over an hour than over 10 minutes.

## Project layout

```
src/cryptosent/
  collectors/   coinmarketcap.py, rss.py, gdelt.py, cryptocompare.py, prices.py (Binance), news_base.py
  sentiment/    prompt.py, gemini.py, lexicon.py (offline), finbert.py (optional), nlp_features.py
  features/     technical.py, sentiment_agg.py, dataset.py (leakage-safe builder + audit)
  models/       registry.py, train.py (protocol), evaluate.py (metrics + backtest), baselines.py
  api/          app.py (FastAPI) + static/index.html (dashboard)
  pipeline.py   tasks shared by CLI, DAGs and services
  streaming.py  Kafka producer + sentiment consumer
  storage.py    MongoStore / LocalStore
  demo.py       synthetic data generator
airflow/dags/   cryptosent_dags.py
docker/         app + airflow images
tests/          unit + end-to-end tests
```

## CLI

```
cryptosent crawl | prices [--backfill-days N] | backfill-news --days N
cryptosent enrich [--backend lexicon] | rescore --from-backend lexicon --limit N
cryptosent dataset --out file.csv | train [--days N] [--models ...] | predict
cryptosent consume | serve | demo
```

Without `MONGO_URI` everything is stored under `./data`, which is handy for
notebooks and coursework.

## Notes and limits

* Site scrapers break when sites change. The CoinMarketCap crawler reads the
  site's JSON content endpoint and falls back to the HTML page. Check it
  against the live site and adjust the parsers in `coinmarketcap.py` if needed.
* Respect each site's terms of use and rate limits. The crawler rotates user
  agents and supports a proxy pool (`HTTP_PROXY_POOL`), but keep request
  rates modest.
* The offline lexicon backend is a fallback, not a replacement for the LLM.
* This is research code, not trading advice.
