"""Window-level aggregation of article sentiment (report sections 3.3, 3.4 and
4.1.1d) plus NLP aggregates (appendix table 7.2) and a few additions:
label shares, a recency-weighted score, news velocity and 6-hour context.

Strict rule: a sample at decision time t only sees articles with
``published_at`` in [t - obs, t). Articles at exactly t are excluded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..sentiment.nlp_features import FEATURE_NAMES as NLP_NAMES

LLM_FEATURES = [
    "sent_n_articles", "sent_mean", "sent_std", "sent_max", "sent_min", "sent_market",
    "sent_conf_mean", "sent_n_manipulation", "sent_frac_bullish", "sent_frac_bearish",
    "sent_frac_neutral", "sent_frac_manipulation", "sent_decay", "sent_velocity",
    "sent_n_articles_6h", "sent_mean_6h",
]
# aggregation rule per NLP feature (report: sums for attention, means for ratios)
NLP_AGG = {
    "pos_ratio": "mean", "neg_ratio": "mean", "compound": "mean", "word_count": "sum",
    "sentence_count": "sum", "avg_sentence_length": "mean", "unique_word_ratio": "mean",
    "exclamation_count": "sum", "question_count": "sum", "btc_mention_count": "sum",
    "price_keyword_count": "sum", "technical_term_ratio": "mean", "hype_count": "sum",
    "ne_person_count": "sum", "ne_org_count": "sum",
}
NLP_FEATURES = [f"nlp_{NLP_AGG[name]}_{name}" for name in NLP_NAMES]
LONG_WINDOW_MIN = 360
DECAY_HALF_LIFE_MIN = 10.0


def _prepare(articles: pd.DataFrame) -> pd.DataFrame:
    if articles.empty:
        return articles
    df = articles.dropna(subset=["sentiment_score"]).copy()
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
    return df.sort_values("published_at").reset_index(drop=True)


def aggregate_windows(articles: pd.DataFrame, decision_times: pd.DatetimeIndex, obs_min: int) -> pd.DataFrame:
    """One row of sentiment + NLP features per decision time."""
    arts = _prepare(articles)
    columns = LLM_FEATURES + NLP_FEATURES + ["last_article_time"]
    if arts.empty:
        out = pd.DataFrame(0.0, index=decision_times, columns=columns)
        out["last_article_time"] = pd.NaT
        return out

    times = arts["published_at"].values.astype("datetime64[ns]")
    score = arts["sentiment_score"].astype(float).to_numpy()
    conf = arts["sentiment_confidence"].fillna(0.5).astype(float).to_numpy()
    label = arts["sentiment_label"].fillna("neutral").to_numpy()
    nlp = {
        name: arts[f"nlp_{name}"].astype(float).fillna(0.0).to_numpy() if f"nlp_{name}" in arts else np.zeros(len(arts))
        for name in NLP_NAMES
    }

    t_ns = decision_times.values.astype("datetime64[ns]")
    hi = np.searchsorted(times, t_ns, side="left")  # strictly before t
    lo = np.searchsorted(times, t_ns - np.timedelta64(obs_min, "m"), side="left")
    lo_recent = np.searchsorted(times, t_ns - np.timedelta64(min(10, obs_min), "m"), side="left")
    lo_long = np.searchsorted(times, t_ns - np.timedelta64(LONG_WINDOW_MIN, "m"), side="left")

    rows = []
    for i in range(len(t_ns)):
        a, b = lo[i], hi[i]
        n = b - a
        row: dict[str, float] = {}
        long_n = b - lo_long[i]
        row["sent_n_articles_6h"] = float(long_n)
        row["sent_mean_6h"] = float(score[lo_long[i]:b].mean()) if long_n else 0.0
        if n == 0:
            row.update({k: 0.0 for k in LLM_FEATURES if k not in row})
            row.update({k: 0.0 for k in NLP_FEATURES})
            row["last_article_time"] = pd.NaT
            rows.append(row)
            continue
        s, c, lab = score[a:b], conf[a:b], label[a:b]
        age_min = (t_ns[i] - times[a:b]) / np.timedelta64(1, "m")
        w_decay = 0.5 ** (age_min / DECAY_HALF_LIFE_MIN)
        row.update({
            "sent_n_articles": float(n),
            "sent_mean": float(s.mean()),
            "sent_std": float(s.std()) if n > 1 else 0.0,
            "sent_max": float(s.max()),
            "sent_min": float(s.min()),
            "sent_market": float((c * s).sum() / c.sum()) if c.sum() > 0 else float(s.mean()),
            "sent_conf_mean": float(c.mean()),
            "sent_n_manipulation": float((lab == "manipulation").sum()),
            "sent_frac_bullish": float((lab == "bullish").mean()),
            "sent_frac_bearish": float((lab == "bearish").mean()),
            "sent_frac_neutral": float((lab == "neutral").mean()),
            "sent_frac_manipulation": float((lab == "manipulation").mean()),
            "sent_decay": float((w_decay * c * s).sum() / (w_decay * c).sum()) if (w_decay * c).sum() > 0 else 0.0,
            "sent_velocity": float((b - lo_recent[i]) / n),
        })
        for name in NLP_NAMES:
            vals = nlp[name][a:b]
            how = NLP_AGG[name]
            row[f"nlp_{how}_{name}"] = float(vals.sum() if how == "sum" else vals.mean())
        row["last_article_time"] = pd.Timestamp(times[b - 1], tz="UTC")
        rows.append(row)

    out = pd.DataFrame(rows, index=decision_times)
    return out[columns]
