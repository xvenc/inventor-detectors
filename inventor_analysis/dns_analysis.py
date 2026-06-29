"""DNS exploratory data analysis functions.

Input DataFrames are produced by ``loaders.load_dns_data()`` and contain the
following columns:

    timestamp, test_id, resolver, query_type, target_host,
    response_time_ms, ttl_s, status, status_code, ip_count, ips, file_date

All public functions accept such a DataFrame and return a new DataFrame with
the analysis results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
# 1. Resolver statistics
# ---------------------------------------------------------------------------

def compute_resolver_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-resolver response-time statistics and success rate.

    Returns a DataFrame with columns:
        resolver, rt_mean, rt_std, rt_min, rt_max, rt_median, rt_p95,
        success_rate, query_count
    sorted by ``rt_mean`` ascending.
    """
    grouped = df.groupby("resolver")

    stats = grouped["response_time_ms"].agg(
        rt_mean="mean",
        rt_std="std",
        rt_min="min",
        rt_max="max",
        rt_median="median",
    )
    stats["rt_p95"] = grouped["response_time_ms"].quantile(0.95)
    stats["success_rate"] = grouped["status"].apply(
        lambda s: (s == "success").mean()
    )
    stats["query_count"] = grouped.size()

    return (
        stats
        .reset_index()
        .sort_values("rt_mean")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 2. Resolver stability (coefficient of variation)
# ---------------------------------------------------------------------------

def compute_resolver_stability(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-resolver coefficient of variation for response time.

    Returns a DataFrame with columns:
        resolver, rt_std, rt_mean, cv
    sorted by ``cv`` ascending (most stable first).
    """
    grouped = df.groupby("resolver")["response_time_ms"]

    stability = pd.DataFrame({
        "rt_std": grouped.std(),
        "rt_mean": grouped.mean(),
    })
    stability["cv"] = stability["rt_std"] / stability["rt_mean"]

    return (
        stability
        .reset_index()
        .sort_values("cv")
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# 3. Anomaly detection via Z-score
# ---------------------------------------------------------------------------

def detect_dns_anomalies(
    df: pd.DataFrame,
    threshold: float = 3.0,
) -> pd.DataFrame:
    """Flag anomalous response times using per-resolver Z-scores.

    Parameters
    ----------
    df : pd.DataFrame
        DNS measurement data.
    threshold : float, optional
        Absolute Z-score above which a measurement is considered anomalous.
        Defaults to 3.0.

    Returns a copy of *df* with two additional columns:
        rt_zscore  – per-resolver Z-score for ``response_time_ms``
        is_anomaly – boolean flag (``|rt_zscore| >= threshold``)
    """
    result = df.copy()

    group_stats = result.groupby("resolver")["response_time_ms"].transform
    mean = group_stats("mean")
    std = group_stats("std")

    # Avoid division by zero when a resolver has a single observation.
    result["rt_zscore"] = np.where(std == 0, 0.0, (result["response_time_ms"] - mean) / std)
    result["is_anomaly"] = result["rt_zscore"].abs() >= threshold

    return result


# ---------------------------------------------------------------------------
# 4. Threshold breaches
# ---------------------------------------------------------------------------

def compute_threshold_breaches(
    df: pd.DataFrame,
    threshold_ms: float = 20,
) -> pd.DataFrame:
    """Compute the fraction of queries exceeding a response-time threshold.

    Parameters
    ----------
    df : pd.DataFrame
        DNS measurement data.
    threshold_ms : float, optional
        Response-time threshold in milliseconds.  Defaults to 20.

    Returns a DataFrame with columns:
        resolver, total_queries, breaches, breach_fraction
    """
    grouped = df.groupby("resolver")

    breaches = pd.DataFrame({
        "total_queries": grouped.size(),
        "breaches": grouped["response_time_ms"].apply(
            lambda s: (s > threshold_ms).sum()
        ),
    })
    breaches["breach_fraction"] = breaches["breaches"] / breaches["total_queries"]

    return breaches.reset_index()


# ---------------------------------------------------------------------------
# 5. Resolver consistency (IP-set Jaccard similarity across days)
# ---------------------------------------------------------------------------

def compute_resolver_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Track IP-set changes per resolver across days.

    For each resolver on each ``file_date`` the function computes:
        * ``unique_ips`` – number of distinct IPs seen that day
        * ``jaccard_prev`` – Jaccard similarity of the IP set compared with
          the previous day (NaN for the first day)

    Returns a DataFrame with columns:
        resolver, file_date, unique_ips, jaccard_prev
    """
    records: list[dict] = []

    for resolver, grp in df.groupby("resolver"):
        # Build a mapping date -> set of IPs.
        daily_ips: dict[str, set] = {}
        for date, day_grp in grp.groupby("file_date"):
            ip_set: set[str] = set()
            for ips_value in day_grp["ips"].dropna():
                if isinstance(ips_value, list):
                    ip_set.update(ips_value)
                elif isinstance(ips_value, str):
                    # Handle comma-separated string representation.
                    ip_set.update(
                        ip.strip() for ip in ips_value.split(",") if ip.strip()
                    )
            daily_ips[date] = ip_set

        sorted_dates = sorted(daily_ips.keys())
        prev_set: set[str] | None = None

        for date in sorted_dates:
            current_set = daily_ips[date]
            if prev_set is not None and (current_set or prev_set):
                intersection = len(current_set & prev_set)
                union = len(current_set | prev_set)
                jaccard = intersection / union if union > 0 else np.nan
            else:
                jaccard = np.nan

            records.append({
                "resolver": resolver,
                "file_date": date,
                "unique_ips": len(current_set),
                "jaccard_prev": jaccard,
            })
            prev_set = current_set

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 6. TTL / response-time correlation
# ---------------------------------------------------------------------------

def compute_ttl_response_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-resolver Pearson correlation between TTL and response time.

    Returns a DataFrame with columns:
        resolver, correlation, p_value
    """
    records: list[dict] = []

    for resolver, grp in df.groupby("resolver"):
        clean = grp[["ttl_s", "response_time_ms"]].dropna()
        if len(clean) < 3:
            # Not enough data points for a meaningful correlation.
            records.append({
                "resolver": resolver,
                "correlation": np.nan,
                "p_value": np.nan,
            })
            continue

        corr, p_val = pearsonr(clean["ttl_s"], clean["response_time_ms"])
        records.append({
            "resolver": resolver,
            "correlation": corr,
            "p_value": p_val,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 7. Combined resolver ranking
# ---------------------------------------------------------------------------

def rank_resolvers(df: pd.DataFrame) -> pd.DataFrame:
    """Rank resolvers by speed, stability, and reliability.

    The overall rank is the mean of the three individual ranks (lower is
    better).

    Returns a DataFrame with columns:
        resolver, rank_speed, rank_stability, rank_reliability, overall_rank
    sorted by ``overall_rank`` ascending.
    """
    grouped = df.groupby("resolver")

    metrics = pd.DataFrame({
        "rt_mean": grouped["response_time_ms"].mean(),
        "cv": grouped["response_time_ms"].std() / grouped["response_time_ms"].mean(),
        "success_rate": grouped["status"].apply(lambda s: (s == "success").mean()),
    })

    # Lower rt_mean is better  -> ascending rank.
    metrics["rank_speed"] = metrics["rt_mean"].rank(method="min")
    # Lower CV is better        -> ascending rank.
    metrics["rank_stability"] = metrics["cv"].rank(method="min")
    # Higher success_rate is better -> descending rank.
    metrics["rank_reliability"] = metrics["success_rate"].rank(
        method="min", ascending=False,
    )

    metrics["overall_rank"] = (
        metrics[["rank_speed", "rank_stability", "rank_reliability"]].mean(axis=1)
    )

    return (
        metrics[["rank_speed", "rank_stability", "rank_reliability", "overall_rank"]]
        .reset_index()
        .sort_values("overall_rank")
        .reset_index(drop=True)
    )
