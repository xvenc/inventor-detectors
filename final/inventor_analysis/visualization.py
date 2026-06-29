"""Reusable visualization functions for Inventor network monitoring data.

All functions create matplotlib/seaborn plots and return the axes object.
An optional ``ax`` parameter allows embedding plots in existing figures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure

sns.set_style("whitegrid")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_ax(ax: matplotlib.axes.Axes | None, figsize: tuple[float, float] = (10, 6)):
    """Return *ax* if provided, otherwise create a new figure and axes."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def _tight(ax: matplotlib.axes.Axes) -> None:
    """Apply tight_layout on the axes' parent figure, ignoring errors."""
    try:
        ax.figure.tight_layout()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 1. RTT distribution histogram
# ---------------------------------------------------------------------------

def plot_rtt_distribution(
    df: pd.DataFrame,
    source_label: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Histogram of ``rtt_avg`` with a vertical median line.

    Parameters
    ----------
    df : DataFrame with an ``rtt_avg`` column.
    source_label : optional label shown in the title.
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)
    data = df["rtt_avg"].dropna()

    sns.histplot(data, bins=50, kde=True, ax=ax, color="steelblue", edgecolor="white")

    median = data.median()
    ax.axvline(median, color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median: {median:.2f} ms")

    title = "RTT Distribution"
    if source_label:
        title += f" ({source_label})"
    ax.set_title(title)
    ax.set_xlabel("RTT Average (ms)")
    ax.set_ylabel("Count")
    ax.legend()
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 2. Host comparison box plot
# ---------------------------------------------------------------------------

def plot_host_comparison(
    df: pd.DataFrame,
    metric: str = "rtt_avg",
    top_n: int = 15,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Box plot of *metric* by ``target_host`` for the *top_n* hosts by count.

    Parameters
    ----------
    df : DataFrame with ``target_host`` and the requested *metric* column.
    metric : numeric column to compare across hosts.
    top_n : number of hosts to include (selected by measurement count).
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax, figsize=(12, 7))

    top_hosts = (
        df["target_host"]
        .value_counts()
        .head(top_n)
        .index.tolist()
    )
    subset = df[df["target_host"].isin(top_hosts)].copy()

    # Order hosts by median metric value for readability
    host_order = (
        subset.groupby("target_host")[metric]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    sns.boxplot(
        data=subset, y="target_host", x=metric, order=host_order,
        palette="viridis", ax=ax, fliersize=2,
    )

    ax.set_title(f"{metric} by Host (top {top_n})")
    ax.set_xlabel(metric)
    ax.set_ylabel("Target Host")
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 3. Daily trend line
# ---------------------------------------------------------------------------

def plot_daily_trend(
    daily_stats: pd.DataFrame,
    metric: str = "rtt_mean",
    label: str = "",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Line plot of a daily metric with +/- 1 std shading.

    Parameters
    ----------
    daily_stats : DataFrame with a date index and columns ``rtt_mean``,
        ``rtt_std``, etc.
    metric : column name for the center line.
    label : optional label appended to the title.
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    dates = daily_stats.index
    mean = daily_stats[metric]

    ax.plot(dates, mean, marker="o", markersize=4, linewidth=1.5, label=metric)

    std_col = metric.replace("_mean", "_std")
    if std_col in daily_stats.columns:
        std = daily_stats[std_col]
        ax.fill_between(dates, mean - std, mean + std, alpha=0.2, label="±1 std")

    title = "Daily Trend"
    if label:
        title += f" ({label})"
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(metric)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 4. Hourly pattern
# ---------------------------------------------------------------------------

def plot_hourly_pattern(
    hourly_stats: pd.DataFrame,
    metric: str = "rtt_mean",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Line plot of *metric* by hour of day (0--23).

    Parameters
    ----------
    hourly_stats : DataFrame with an ``hour`` column (0--23) and the
        requested *metric*.
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    ax.plot(hourly_stats["hour"], hourly_stats[metric],
            marker="o", linewidth=1.5, color="teal")

    ax.set_title("Hourly Pattern")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel(metric)
    ax.set_xticks(range(0, 24))
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 5. RTT heatmap
# ---------------------------------------------------------------------------

def plot_rtt_heatmap(
    df: pd.DataFrame,
    top_n: int = 10,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Heatmap of mean ``rtt_avg`` with hosts on the y-axis and dates on x.

    Parameters
    ----------
    df : DataFrame with ``target_host``, ``rtt_avg``, and ``timestamp``
        (or ``file_date``) columns.
    top_n : number of hosts to include (by measurement count).
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax, figsize=(14, 6))

    top_hosts = df["target_host"].value_counts().head(top_n).index.tolist()
    subset = df[df["target_host"].isin(top_hosts)].copy()

    if "file_date" in subset.columns:
        subset["date"] = pd.to_datetime(subset["file_date"])
    else:
        subset["date"] = subset["timestamp"].dt.date

    pivot = subset.pivot_table(
        values="rtt_avg", index="target_host", columns="date", aggfunc="mean",
    )

    sns.heatmap(
        pivot, cmap="YlOrRd", annot=False, fmt=".1f",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Mean RTT (ms)"},
    )

    ax.set_title(f"RTT Heatmap (top {top_n} hosts)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Target Host")
    ax.tick_params(axis="x", rotation=45)
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 6. Anomaly timeline
# ---------------------------------------------------------------------------

def plot_anomaly_timeline(
    df: pd.DataFrame,
    host: str,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Scatter plot of RTT over time with anomalies highlighted in red.

    Parameters
    ----------
    df : DataFrame with ``target_host``, ``timestamp``, ``rtt_avg``, and
        ``is_anomaly`` columns (e.g. output of
        :func:`ping_analysis.detect_anomalies_zscore`).
    host : the host to plot.
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    host_df = df[df["target_host"] == host].copy()
    normal = host_df[~host_df["is_anomaly"]]
    anomalies = host_df[host_df["is_anomaly"]]

    ax.scatter(normal["timestamp"], normal["rtt_avg"],
               s=12, alpha=0.5, label="Normal", color="steelblue")
    ax.scatter(anomalies["timestamp"], anomalies["rtt_avg"],
               s=30, alpha=0.8, label="Anomaly", color="red", zorder=5)

    ax.set_title(f"Anomaly Timeline: {host}")
    ax.set_xlabel("Time")
    ax.set_ylabel("RTT Average (ms)")
    ax.legend()
    ax.tick_params(axis="x", rotation=45)
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 7. DNS resolver comparison
# ---------------------------------------------------------------------------

def plot_dns_resolver_comparison(
    resolver_stats: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Horizontal bar chart of mean response time per resolver.

    Parameters
    ----------
    resolver_stats : DataFrame with ``resolver`` and ``response_time_mean``
        columns (or similar; the first numeric column after ``resolver``
        is used when ``response_time_mean`` is absent).
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    stats = resolver_stats.sort_values("response_time_mean", ascending=True)

    ax.barh(stats["resolver"].astype(str), stats["response_time_mean"],
            color=sns.color_palette("coolwarm", n_colors=len(stats)))

    ax.set_title("DNS Resolver Comparison")
    ax.set_xlabel("Mean Response Time (ms)")
    ax.set_ylabel("Resolver")
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 8. DNS time series (small multiples)
# ---------------------------------------------------------------------------

def plot_dns_time_series(
    df: pd.DataFrame,
    resample: str = "5min",
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Per-resolver resampled median response time as small multiples.

    Parameters
    ----------
    df : DNS DataFrame with ``timestamp``, ``resolver``, and
        ``response_time_ms`` columns.
    resample : pandas resample frequency string.
    ax : ignored -- a new figure is always created for the subplot grid.
        The first subplot axes is returned.

    Returns
    -------
    The first axes of the small-multiple grid.
    """
    resolvers = df["resolver"].dropna().unique()
    n = len(resolvers)
    if n == 0:
        ax = _get_ax(ax)
        ax.set_title("DNS Time Series (no data)")
        return ax

    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, resolver in enumerate(sorted(resolvers)):
        cur_ax = axes[i]
        rdf = df[df["resolver"] == resolver].copy()
        rdf = rdf.set_index("timestamp").sort_index()
        resampled = rdf["response_time_ms"].resample(resample).median()

        cur_ax.plot(resampled.index, resampled.values, linewidth=1, color="teal")
        cur_ax.set_ylabel("Median RT (ms)")
        cur_ax.set_title(f"Resolver: {resolver}")

    axes[-1].set_xlabel("Time")
    fig.suptitle("DNS Response Time by Resolver", fontsize=14, y=1.01)
    fig.tight_layout()
    return axes[0]


# ---------------------------------------------------------------------------
# 9. Hop latency build-up
# ---------------------------------------------------------------------------

def plot_hop_latency_buildup(
    hop_stats: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Line plot of mean RTT by hop number with +/- 1 std shading.

    Parameters
    ----------
    hop_stats : DataFrame with ``hop_number``, ``hop_rtt_mean``, and
        ``hop_rtt_std`` columns (or raw hop data with ``hop_number`` and
        ``hop_rtt``).
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    # Support both pre-aggregated and raw hop data
    if "hop_rtt_mean" in hop_stats.columns:
        hops = hop_stats.sort_values("hop_number")
        mean_col = "hop_rtt_mean"
        std_col = "hop_rtt_std"
    else:
        hops = (
            hop_stats.groupby("hop_number")["hop_rtt"]
            .agg(["mean", "std"])
            .reset_index()
            .rename(columns={"mean": "hop_rtt_mean", "std": "hop_rtt_std"})
        )
        mean_col = "hop_rtt_mean"
        std_col = "hop_rtt_std"

    ax.plot(hops["hop_number"], hops[mean_col],
            marker="o", linewidth=1.5, color="darkorange", label="Mean RTT")
    ax.fill_between(
        hops["hop_number"],
        hops[mean_col] - hops[std_col].fillna(0),
        hops[mean_col] + hops[std_col].fillna(0),
        alpha=0.2, color="darkorange", label="±1 std",
    )

    ax.set_title("Hop Latency Build-up")
    ax.set_xlabel("Hop Number")
    ax.set_ylabel("RTT (ms)")
    ax.legend()
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 10. Diagnosis distribution
# ---------------------------------------------------------------------------

_SEVERITY_COLORS = {
    "CONGESTION": "gold",
    "PATH_CHANGE": "orange",
    "LINK_SATURATION": "red",
    "INFRASTRUCTURE_FAILURE": "darkred",
}


def plot_diagnosis_distribution(
    diagnoses_df: pd.DataFrame,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Horizontal bar chart of diagnosis type counts, color-coded by severity.

    Parameters
    ----------
    diagnoses_df : DataFrame with a ``diagnosis`` (or ``diagnosis_type``)
        column whose values match the severity colour map.
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    col = "diagnosis" if "diagnosis" in diagnoses_df.columns else "diagnosis_type"
    counts = diagnoses_df[col].value_counts()

    colors = [_SEVERITY_COLORS.get(d, "grey") for d in counts.index]

    ax.barh(counts.index.astype(str), counts.values, color=colors)
    ax.set_title("Diagnosis Distribution")
    ax.set_xlabel("Count")
    ax.set_ylabel("Diagnosis")
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 11. Generic correlation scatter
# ---------------------------------------------------------------------------

def plot_correlation_scatter(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    ax: matplotlib.axes.Axes | None = None,
) -> matplotlib.axes.Axes:
    """Scatter plot with optional colour coding and Pearson-r annotation.

    Parameters
    ----------
    df : source DataFrame.
    x_col, y_col : columns for x and y axes.
    color_col : optional categorical column used for colour coding.
    xlabel, ylabel, title : optional axis / title overrides.
    ax : matplotlib axes to draw on; a new figure is created when *None*.
    """
    ax = _get_ax(ax)

    plot_df = df[[x_col, y_col]].dropna()
    if color_col is not None and color_col in df.columns:
        plot_df = plot_df.join(df[color_col])

    if color_col and color_col in plot_df.columns:
        sns.scatterplot(data=plot_df, x=x_col, y=y_col, hue=color_col,
                        alpha=0.6, ax=ax, s=20)
    else:
        ax.scatter(plot_df[x_col], plot_df[y_col],
                   alpha=0.5, s=15, color="steelblue")

    # Pearson correlation annotation
    if len(plot_df) > 2:
        corr = plot_df[x_col].corr(plot_df[y_col])
        ax.annotate(
            f"r = {corr:.3f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=11, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8),
        )

    ax.set_xlabel(xlabel or x_col)
    ax.set_ylabel(ylabel or y_col)
    ax.set_title(title or f"{x_col} vs {y_col}")
    _tight(ax)
    return ax


# ---------------------------------------------------------------------------
# 12. Summary dashboard
# ---------------------------------------------------------------------------

def create_summary_dashboard(
    ping_df: pd.DataFrame,
    dns_df: pd.DataFrame | None = None,
    trace_summary: pd.DataFrame | None = None,
) -> matplotlib.figure.Figure:
    """Create a 3x3 grid dashboard combining key network visualizations.

    Parameters
    ----------
    ping_df : ping DataFrame from :func:`loaders.load_ping_data`.
    dns_df : optional DNS DataFrame from :func:`loaders.load_dns_data`.
    trace_summary : optional traceroute summary DataFrame.

    Returns
    -------
    The :class:`matplotlib.figure.Figure` containing all subplots.
    """
    fig, axes = plt.subplots(3, 3, figsize=(20, 16))

    # -- Row 0 ---------------------------------------------------------------

    # (0,0) RTT distribution
    plot_rtt_distribution(ping_df, ax=axes[0, 0])

    # (0,1) Daily RTT trend
    if "timestamp" in ping_df.columns:
        work = ping_df.copy()
        work["date"] = work["timestamp"].dt.date
        daily = work.groupby("date").agg(
            rtt_mean=("rtt_avg", "mean"),
            rtt_std=("rtt_avg", "std"),
        )
        plot_daily_trend(daily, ax=axes[0, 1])
    else:
        axes[0, 1].set_title("Daily Trend (no timestamp)")

    # (0,2) Jitter comparison (top 10 hosts by count)
    top_hosts = ping_df["target_host"].value_counts().head(10).index.tolist()
    jitter_subset = ping_df[ping_df["target_host"].isin(top_hosts)]
    host_order = (
        jitter_subset.groupby("target_host")["jitter"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )
    sns.boxplot(
        data=jitter_subset, y="target_host", x="jitter",
        order=host_order, palette="Oranges_r", ax=axes[0, 2], fliersize=2,
    )
    axes[0, 2].set_title("Jitter by Host (top 10)")
    axes[0, 2].set_xlabel("Jitter (ms)")
    axes[0, 2].set_ylabel("")

    # -- Row 1 ---------------------------------------------------------------

    # (1,0) Host performance bar chart (mean RTT, top 10)
    host_perf = (
        ping_df.groupby("target_host")["rtt_avg"]
        .mean()
        .sort_values()
        .tail(10)
    )
    axes[1, 0].barh(host_perf.index.astype(str), host_perf.values,
                     color=sns.color_palette("RdYlGn_r", n_colors=len(host_perf)))
    axes[1, 0].set_title("Slowest Hosts (mean RTT)")
    axes[1, 0].set_xlabel("Mean RTT (ms)")

    # (1,1) Packet loss per host
    loss = (
        ping_df.groupby("target_host")["pkts_lost"]
        .mean()
        .sort_values()
        .tail(10)
    )
    axes[1, 1].barh(loss.index.astype(str), loss.values,
                     color=sns.color_palette("Reds", n_colors=len(loss)))
    axes[1, 1].set_title("Packet Loss by Host (top 10)")
    axes[1, 1].set_xlabel("Mean Packets Lost")

    # (1,2) RTT vs Jitter scatter
    if "jitter" in ping_df.columns:
        plot_correlation_scatter(
            ping_df, "rtt_avg", "jitter",
            title="RTT vs Jitter", ax=axes[1, 2],
        )
    else:
        axes[1, 2].set_title("RTT vs Jitter (no data)")

    # -- Row 2 ---------------------------------------------------------------

    # (2,0) DNS resolver comparison (if available)
    if dns_df is not None and not dns_df.empty:
        resolver_stats = (
            dns_df.groupby("resolver")["response_time_ms"]
            .mean()
            .reset_index()
            .rename(columns={"response_time_ms": "response_time_mean"})
        )
        plot_dns_resolver_comparison(resolver_stats, ax=axes[2, 0])
    else:
        axes[2, 0].text(
            0.5, 0.5, "No DNS data", transform=axes[2, 0].transAxes,
            ha="center", va="center", fontsize=14, color="grey",
        )
        axes[2, 0].set_title("DNS Resolver Comparison")

    # (2,1) Traceroute packet loss (if available)
    if trace_summary is not None and not trace_summary.empty:
        trace_loss = (
            trace_summary.groupby("target_host")["packet_loss"]
            .mean()
            .sort_values()
            .tail(10)
        )
        axes[2, 1].barh(
            trace_loss.index.astype(str), trace_loss.values,
            color=sns.color_palette("OrRd", n_colors=len(trace_loss)),
        )
        axes[2, 1].set_title("Traceroute Packet Loss by Host")
        axes[2, 1].set_xlabel("Mean Packet Loss (%)")
    else:
        axes[2, 1].text(
            0.5, 0.5, "No traceroute data", transform=axes[2, 1].transAxes,
            ha="center", va="center", fontsize=14, color="grey",
        )
        axes[2, 1].set_title("Traceroute Packet Loss")

    # (2,2) Measurement count over time
    if "timestamp" in ping_df.columns:
        work = ping_df.copy()
        work["date"] = work["timestamp"].dt.date
        counts = work.groupby("date").size()
        axes[2, 2].bar(counts.index, counts.values, color="steelblue", alpha=0.7)
        axes[2, 2].set_title("Measurements per Day")
        axes[2, 2].set_xlabel("Date")
        axes[2, 2].set_ylabel("Count")
        axes[2, 2].tick_params(axis="x", rotation=45)
    else:
        axes[2, 2].set_title("Measurements per Day (no timestamp)")

    fig.suptitle("Network Monitoring Summary Dashboard", fontsize=16, y=1.01)
    fig.tight_layout()
    return fig
