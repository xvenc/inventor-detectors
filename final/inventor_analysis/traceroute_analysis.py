"""Traceroute data analysis functions.

Expects input DataFrames from loaders.load_traceroute_data() which returns:
    summaries_df: timestamp, test_id, target_host, ip_address, min_hops,
                  max_hops, path_stability, packet_loss, file_date
    hops_df:      timestamp, target_host, run_id, hop_number, hop_ip, hop_rtt
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hop_statistics(hops_df: pd.DataFrame) -> pd.DataFrame:
    """Per-hop-position latency statistics showing RTT buildup across hops."""
    stats = hops_df.groupby("hop_number").agg(
        mean_rtt=("hop_rtt", "mean"),
        std_rtt=("hop_rtt", "std"),
        min_rtt=("hop_rtt", "min"),
        max_rtt=("hop_rtt", "max"),
        p95_rtt=("hop_rtt", lambda x: np.percentile(x.dropna(), 95)),
        sample_count=("hop_rtt", "size"),
    )
    return stats.reset_index()


def compute_gateway_performance(
    hops_df: pd.DataFrame, min_occurrences: int = 5,
) -> pd.DataFrame:
    """Per-gateway-IP latency profile, filtered to frequently seen gateways."""
    stats = hops_df.groupby("hop_ip").agg(
        mean_rtt=("hop_rtt", "mean"),
        std_rtt=("hop_rtt", "std"),
        occurrence_count=("hop_rtt", "size"),
    )
    stats = stats[stats["occurrence_count"] >= min_occurrences]
    return stats.sort_values("occurrence_count", ascending=False).reset_index()


def compute_incremental_latency(hops_df: pd.DataFrame) -> pd.DataFrame:
    """Add per-hop incremental RTT (diff between consecutive hops).

    Traceroute sessions are identified by (timestamp, target_host).
    Within each session hops are sorted by hop_number and the incremental
    RTT is computed as the difference from the previous hop.  The first hop
    in each session gets NaN.
    """
    result = hops_df.copy()
    result = result.sort_values(["timestamp", "target_host", "hop_number"])
    result["incremental_rtt"] = result.groupby(
        ["timestamp", "target_host"],
    )["hop_rtt"].diff()
    return result


def get_path_signatures(hops_df: pd.DataFrame) -> pd.DataFrame:
    """Build a path signature string for each traceroute session.

    Returns a DataFrame with columns: timestamp, target_host, path_signature.
    The path_signature is the ordered hop IPs joined with ' -> '.
    """
    ordered = hops_df.sort_values(
        ["timestamp", "target_host", "hop_number"],
    )
    signatures = (
        ordered.groupby(["timestamp", "target_host"])["hop_ip"]
        .apply(lambda ips: " -> ".join(ips.astype(str)))
        .reset_index()
        .rename(columns={"hop_ip": "path_signature"})
    )
    return signatures


def compute_path_stability(summaries_df: pd.DataFrame) -> pd.DataFrame:
    """Per-target routing stability metrics.

    Returns a DataFrame with columns: target_host, mean_path_stability,
    stability_variance, hop_count_variance.  High stability_variance
    indicates route-volatile destinations.
    """
    stats = summaries_df.groupby("target_host").agg(
        mean_path_stability=("path_stability", "mean"),
        stability_variance=("path_stability", "var"),
        hop_count_variance=("max_hops", "var"),
    )
    return stats.reset_index()


def build_hop_baselines(hops_df: pd.DataFrame) -> pd.DataFrame:
    """Historical RTT baselines per (target, hop_number, hop_ip).

    Used as the reference for root-cause localization.
    """
    baselines = hops_df.groupby(
        ["target_host", "hop_number", "hop_ip"],
    ).agg(
        mean_rtt=("hop_rtt", "mean"),
        std_rtt=("hop_rtt", "std"),
        median_rtt=("hop_rtt", "median"),
        p95_rtt=("hop_rtt", lambda x: np.percentile(x.dropna(), 95)),
        sample_count=("hop_rtt", "size"),
    )
    return baselines.reset_index()


def localize_root_cause(
    current_hops: list[dict],
    baselines_df: pd.DataFrame,
    threshold_std: float = 3.0,
    min_deviation_ms: float = 10,
) -> dict:
    """Identify the hop most responsible for a latency anomaly.

    Parameters
    ----------
    current_hops
        List of dicts with keys ``hop_num``, ``ip``, ``rtt``.
    baselines_df
        Baselines built by :func:`build_hop_baselines`.
    threshold_std
        Number of standard deviations above the mean to flag a hop.
    min_deviation_ms
        Minimum absolute deviation (ms) required to flag a hop.

    Returns
    -------
    dict
        ``problem_detected`` (bool), and when True: ``root_cause_hop``,
        ``root_cause_ip``, ``deviation_ms``, ``deviation_std``,
        ``confidence``, ``explanation``.
    """
    worst_deviation_std = 0.0
    worst: dict | None = None

    for hop in current_hops:
        hop_num = hop["hop_num"]
        ip = hop["ip"]
        rtt = hop["rtt"]

        match = baselines_df[
            (baselines_df["hop_number"] == hop_num)
            & (baselines_df["hop_ip"] == ip)
        ]
        if match.empty:
            continue

        baseline = match.iloc[0]
        mean = baseline["mean_rtt"]
        std = baseline["std_rtt"]

        deviation_ms = rtt - mean

        if std > 0:
            deviation_std = deviation_ms / std
        else:
            # Zero variance baseline -- any deviation is significant
            deviation_std = float("inf") if deviation_ms > 0 else 0.0

        if (
            deviation_std > worst_deviation_std
            and deviation_std >= threshold_std
            and deviation_ms >= min_deviation_ms
        ):
            worst_deviation_std = deviation_std
            sample_count = int(baseline["sample_count"])
            confidence = min(1.0, sample_count / 50)

            worst = {
                "problem_detected": True,
                "root_cause_hop": hop_num,
                "root_cause_ip": ip,
                "deviation_ms": round(deviation_ms, 2),
                "deviation_std": round(deviation_std, 2),
                "confidence": round(confidence, 2),
                "explanation": (
                    f"Hop {hop_num} ({ip}) is {deviation_std:.1f} std devs "
                    f"above baseline (current {rtt:.1f} ms vs mean "
                    f"{mean:.1f} ms, based on {sample_count} samples)"
                ),
            }

    if worst is not None:
        return worst

    return {
        "problem_detected": False,
        "root_cause_hop": None,
        "root_cause_ip": None,
        "deviation_ms": 0.0,
        "deviation_std": 0.0,
        "confidence": 0.0,
        "explanation": "All hops within normal baseline range",
    }


def get_network_segments(
    hops_df: pd.DataFrame, n_hops: int = 3,
) -> pd.DataFrame:
    """Extract the first *n_hops* as a network-segment identifier.

    Returns a DataFrame with columns: timestamp, target_host,
    network_segment (string of first N hop IPs separated by ' -> ').
    """
    ordered = hops_df.sort_values(
        ["timestamp", "target_host", "hop_number"],
    )
    first_n = ordered.groupby(["timestamp", "target_host"]).head(n_hops)
    segments = (
        first_n.groupby(["timestamp", "target_host"])["hop_ip"]
        .apply(lambda ips: " -> ".join(ips.astype(str)))
        .reset_index()
        .rename(columns={"hop_ip": "network_segment"})
    )
    return segments
