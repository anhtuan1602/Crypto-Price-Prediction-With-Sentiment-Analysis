# Crypto Price Prediction With Sentiment Analysis

An end-to-end web-mining system that crawls crypto news, scores each article
with an LLM (Gemini 2.5 Flash), aggregates sentiment into time windows and
predicts short-term Bitcoin price moves. It follows the design in the Group 08
technical report and fixes the issues that would make its results hard to
defend (see [What changed from the report](#what-changed-from-the-report)).

```
 CoinMarketCap ─┐                                    ┌─> MongoDB (articles, sentiment, nlp)
 RSS feeds ─────┼─ Airflow cmc_news_10m ─> Kafka ────┤
 CryptoCompare ─┘   (crawl, clean, dedup)  news.raw   └─ sentiment-consumer (Gemini + NLP) ─> news.scored
 Binance 1m ────── Airflow btc_prices_5m ─────────────> MongoDB (prices)
                   Airflow train_daily ─> leakage-safe dataset ─> walk-forward eval ─> MLflow + model
                   Airflow predict_10m ─> predictions ─> FastAPI ─> dashboard
```

## Quick start (offline demo, no services, about 1 minute)

```bash
pip install -e ".[dev]"
cryptosent demo --days 30      # synthetic prices + news, full train/eval
cryptosent serve               # dashboard on http://localhost:8000
pytest                         # 58 tests
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

Then, once, trigger the **`backfill_history`** DAG in Airflow (default 90 days).
It loads 1-minute Binance candles and historical CryptoCompare news, then
scores them. After that the scheduled DAGs keep everything current:

| DAG | Schedule | What it does |
|---|---|---|
| `cmc_news_10m` | every 10 min | crawl CoinMarketCap + RSS, dedupe, store, publish to Kafka |
| `btc_prices_5m` | every 5 min | sync Binance 1m candles |
| `predict_10m` | every 10 min | predict the next window with the latest model |
| `sentiment_catchup_1h` | hourly | score anything the Kafka consumer missed, upgrade some lexicon scores to Gemini |
| `train_daily` | 02:00 UTC | backfill 2 days of news, retrain on the last 180 days, log to MLflow |
| `backfill_history` | manual | initial history load |

### Gemini on the free tier

* The client scores `GEMINI_BATCH_SIZE` articles (default 10) per request
  and paces calls to `GEMINI_RPM` (default 10). It stores each result, so no
  article is scored twice.
* If a model answers 404, the client switches to `gemini-flash-latest`, then
  to other Flash models your key lists, trying at most 3 models.
* A 400 is retried once with a minimal request. A rejected key, an unknown
  model or a request that still fails gives **one** clear error, not a retry
  storm. In `auto` mode the rest of the run then uses the offline lexicon
  (upgrade later with `cryptosent rescore`).
* When the daily quota runs out, scoring stops and the remaining articles
  wait for the next run. The API key goes in a request header and is
  redacted from logs.
* A 90-day backfill is tens of thousands of articles, too many for a free
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
| CoinMarketCap only, 20 days | + RSS + CryptoCompare backfill | months of history and redundancy |
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
  collectors/   coinmarketcap.py, rss.py, cryptocompare.py, prices.py (Binance), news_base.py
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
