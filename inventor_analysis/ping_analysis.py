"""Ping / RTT exploratory data analysis functions.

Expects input DataFrames from loaders.load_ping_data() with columns:
    timestamp, target_host, rtt_avg, rtt_min, rtt_max, rtt_stddev,
    jitter, pkts_sent, pkts_received, pkts_lost, file_date
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_percentile(series, pct):
    vals = series.dropna().values
    if len(vals) == 0:
        return np.nan
    return np.percentile(vals, pct)


def compute_host_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-host aggregation of RTT, jitter, packet-loss, and measurement count."""
    grouped = df.groupby("target_host")["rtt_avg"]
    stats = pd.DataFrame({
        "rtt_mean": grouped.mean(),
        "rtt_std": grouped.std(),
        "rtt_median": grouped.median(),
        "rtt_p95": grouped.apply(lambda x: _safe_percentile(x, 95)),
        "rtt_p99": grouped.apply(lambda x: _safe_percentile(x, 99)),
        "jitter_mean": df.groupby("target_host")["jitter"].mean(),
        "packet_loss_mean": df.groupby("target_host")["pkts_lost"].mean(),
        "measurement_count": grouped.size(),
    })
    return stats.sort_values("rtt_mean").reset_index()


def compute_daily_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Daily aggregation of RTT, jitter, and packet-loss metrics."""
    work = df.copy()
    work["date"] = work["timestamp"].dt.date

    trends = work.groupby("date").agg(
        rtt_mean=("rtt_avg", "mean"),
        rtt_std=("rtt_avg", "std"),
        rtt_min=("rtt_min", "min"),
        rtt_max=("rtt_max", "max"),
        jitter_mean=("jitter", "mean"),
        packet_loss_mean=("pkts_lost", "mean"),
    )
    return trends.reset_index()


def compute_hourly_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """Hourly (0-23) aggregation to reveal diurnal patterns."""
    work = df.copy()
    work["hour"] = work["timestamp"].dt.hour

    patterns = work.groupby("hour").agg(
        rtt_mean=("rtt_avg", "mean"),
        rtt_std=("rtt_avg", "std"),
        jitter_mean=("jitter", "mean"),
    )
    return patterns.reset_index()


def detect_anomalies_zscore(
    df: pd.DataFrame, threshold: float = 3.0
) -> pd.DataFrame:
    """Per-host Z-score anomaly detection on rtt_avg.

    Adds ``rtt_zscore`` and ``is_anomaly`` columns to a copy of *df*.
    A measurement is anomalous when its absolute Z-score exceeds *threshold*.
    """
    result = df.copy()

    host_stats = result.groupby("target_host")["rtt_avg"].transform
    mean = host_stats("mean")
    std = host_stats("std")

    # Hosts with zero std (single measurement or constant RTT) get zscore 0
    result["rtt_zscore"] = np.where(std > 0, (result["rtt_avg"] - mean) / std, 0.0)
    result["is_anomaly"] = result["rtt_zscore"].abs() > threshold

    return result


def compute_loss_rate(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``loss_rate`` column: (pkts_sent - pkts_received) / pkts_sent."""
    result = df.copy()
    result["loss_rate"] = np.where(
        result["pkts_sent"] > 0,
        (result["pkts_sent"] - result["pkts_received"]) / result["pkts_sent"],
        np.nan,
    )
    return result


def rank_hosts(
    df: pd.DataFrame,
    metric: str = "rtt_avg",
    top_n: int = 10,
    ascending: bool = True,
) -> list[str]:
    """Return *top_n* host names sorted by mean of *metric*."""
    ranked = (
        df.groupby("target_host")[metric]
        .mean()
        .sort_values(ascending=ascending)
        .head(top_n)
    )
    return ranked.index.tolist()


def compute_reliability_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-host reliability breakdown.

    Columns:
        success_rate           -- pkts_received / pkts_sent
        pct_rtt_above_threshold -- fraction of measurements with rtt_avg > 150 ms
        pct_jitter_above_threshold -- fraction with jitter > 100 ms
        pct_loss_nonzero       -- fraction with any packet loss
    """

    def _reliability(group: pd.DataFrame) -> pd.Series:
        total_sent = group["pkts_sent"].sum()
        success_rate = (
            group["pkts_received"].sum() / total_sent if total_sent > 0 else np.nan
        )
        n = len(group)
        return pd.Series(
            {
                "success_rate": success_rate,
                "pct_rtt_above_threshold": (group["rtt_avg"] > 150).sum() / n,
                "pct_jitter_above_threshold": (group["jitter"] > 100).sum() / n,
                "pct_loss_nonzero": (group["pkts_lost"] > 0).sum() / n,
            }
        )

    scores = df.groupby("target_host").apply(_reliability, include_groups=False)
    return scores.reset_index()
