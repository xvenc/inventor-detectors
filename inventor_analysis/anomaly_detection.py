"""ML-based anomaly detection models for network monitoring.

Expects input DataFrames from loaders.load_ping_data() with columns:
    timestamp, target_host, rtt_avg, rtt_min, rtt_max, rtt_stddev,
    jitter, pkts_sent, pkts_received, pkts_lost
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdaptiveBaselineDetector:
    """Ultra-strict adaptive baseline anomaly detector targeting Precision > 0.6.

    Learns per-destination statistical baselines from training data and applies
    multi-condition detection rules that require several indicators to fire
    simultaneously, minimising false positives.
    """

    def __init__(self) -> None:
        self.baselines: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data: pd.DataFrame) -> None:
        """Learn per-destination baselines from historical data.

        For each ``target_host`` the following statistics are stored:
        rtt_p999, jitter_p99, rtt_median, rtt_p95, rtt_mad, n_samples.
        """
        for host, group in data.groupby("target_host"):
            rtt = group["rtt_avg"].dropna().values
            jitter = group["jitter"].dropna().values

            if len(rtt) == 0:
                continue

            rtt_median = float(np.median(rtt))
            self.baselines[host] = {
                "rtt_p999": float(np.percentile(rtt, 99.9)),
                "jitter_p99": float(np.percentile(jitter, 99)) if len(jitter) > 0 else 0.0,
                "rtt_median": rtt_median,
                "rtt_p95": float(np.percentile(rtt, 95)),
                "rtt_mad": float(np.median(np.abs(rtt - rtt_median))),
                "n_samples": len(rtt),
            }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self, data: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Classify each row as anomalous or normal.

        Returns
        -------
        predictions : np.ndarray of bool
            ``True`` for anomalous rows.
        severity : np.ndarray of float
            A 0-1 severity score for each row.
        reasons : list[str]
            Human-readable reason tag per row.  One of ``'rtt_jitter'``,
            ``'rtt_loss'``, ``'extreme_rtt'``, ``'catastrophic'``, or
            ``'none'``.
        """
        n = len(data)
        predictions = np.zeros(n, dtype=bool)
        severity = np.zeros(n, dtype=float)
        reasons: list[str] = ["none"] * n

        for idx in range(n):
            row = data.iloc[idx]
            host = row.get("target_host")
            baseline = self.baselines.get(host)

            if baseline is None:
                continue

            rtt = row.get("rtt_avg")
            jitter_val = row.get("jitter")
            loss = row.get("pkts_lost")
            sent = row.get("pkts_sent")

            if rtt is None or np.isnan(rtt):
                continue

            loss_pct = (loss / sent * 100) if (sent and sent > 0) else 0.0

            # Individual condition flags
            extreme_rtt = rtt > baseline["rtt_p999"] and rtt > 100
            high_jitter = (
                jitter_val is not None
                and not np.isnan(jitter_val)
                and jitter_val > baseline["jitter_p99"]
            )
            severe_loss = loss_pct > 30

            mad = baseline["rtt_mad"]
            massive_deviation = (
                mad > 0 and abs(rtt - baseline["rtt_median"]) > 10 * mad
            )

            catastrophic = loss_pct > 50 and rtt > baseline["rtt_p95"]

            # Combined rules
            is_anomaly = False
            reason = "none"

            if extreme_rtt and high_jitter:
                is_anomaly = True
                reason = "rtt_jitter"
            elif extreme_rtt and severe_loss:
                is_anomaly = True
                reason = "rtt_loss"
            elif extreme_rtt and massive_deviation:
                is_anomaly = True
                reason = "extreme_rtt"
            elif catastrophic:
                is_anomaly = True
                reason = "catastrophic"

            if is_anomaly:
                predictions[idx] = True
                reasons[idx] = reason

                # Severity: blend of normalised RTT excess and loss percentage
                rtt_excess = (rtt - baseline["rtt_median"]) / baseline["rtt_median"] if baseline["rtt_median"] > 0 else 0.0
                severity[idx] = float(np.clip(
                    0.5 * min(rtt_excess / 10.0, 1.0) + 0.5 * min(loss_pct / 100.0, 1.0),
                    0.0,
                    1.0,
                ))

        return predictions, severity, reasons


class CorrelatedDegradationDetector:
    """Detects when multiple destinations degrade simultaneously.

    Network-wide incidents are characterised by anomalies hitting several
    destinations within the same short time window.
    """

    def __init__(self, time_window_minutes: int = 5, min_affected: int = 3) -> None:
        self.time_window_minutes = time_window_minutes
        self.min_affected = min_affected

    def detect(
        self, data: pd.DataFrame, baseline_detector: AdaptiveBaselineDetector
    ) -> pd.DataFrame:
        """Identify correlated degradation windows.

        Parameters
        ----------
        data : pd.DataFrame
            Ping data with ``timestamp`` and ``target_host``.
        baseline_detector : AdaptiveBaselineDetector
            A trained baseline detector used to label individual anomalies.

        Returns
        -------
        pd.DataFrame
            Columns: window_start, window_end, affected_count,
            affected_destinations, severity.
        """
        predictions, severity, _ = baseline_detector.predict(data)
        anomalous = data.loc[predictions].copy()

        if anomalous.empty:
            return pd.DataFrame(columns=[
                "window_start", "window_end", "affected_count",
                "affected_destinations", "severity",
            ])

        anomalous = anomalous.sort_values("timestamp").reset_index(drop=True)
        anomalous["_severity"] = severity[predictions]

        window_delta = pd.Timedelta(minutes=self.time_window_minutes)

        # Build non-overlapping windows starting from the earliest anomaly
        min_ts = anomalous["timestamp"].min()
        max_ts = anomalous["timestamp"].max()

        windows: list[dict] = []
        window_start = min_ts

        while window_start <= max_ts:
            window_end = window_start + window_delta
            mask = (anomalous["timestamp"] >= window_start) & (
                anomalous["timestamp"] < window_end
            )
            window_data = anomalous.loc[mask]

            if not window_data.empty:
                affected = window_data["target_host"].unique()
                if len(affected) >= self.min_affected:
                    windows.append({
                        "window_start": window_start,
                        "window_end": window_end,
                        "affected_count": len(affected),
                        "affected_destinations": list(affected),
                        "severity": float(window_data["_severity"].mean()),
                    })

            window_start = window_end

        if not windows:
            return pd.DataFrame(columns=[
                "window_start", "window_end", "affected_count",
                "affected_destinations", "severity",
            ])

        return pd.DataFrame(windows)


class LatencyShiftDetector:
    """CUSUM-based latency shift detection.

    Monitors the cumulative sum of deviations from a running baseline to
    identify sustained shifts in round-trip time for each destination.
    """

    def __init__(
        self,
        sensitivity: float = 5.0,
        drift: float = 1.0,
        baseline_samples: int = 50,
    ) -> None:
        self.sensitivity = sensitivity
        self.drift = drift
        self.baseline_samples = baseline_samples

    def detect(self, data: pd.DataFrame) -> pd.DataFrame:
        """Run CUSUM shift detection per destination.

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, destination, new_rtt, baseline_rtt,
            rtt_change_pct.  One row per detected shift point.
        """
        results: list[dict] = []

        for host, group in data.groupby("target_host"):
            group = group.sort_values("timestamp").reset_index(drop=True)
            rtt = group["rtt_avg"].values
            timestamps = group["timestamp"].values

            n = len(rtt)
            if n < self.baseline_samples + 1:
                continue

            baseline_window = rtt[: self.baseline_samples]
            baseline_mean = float(np.nanmean(baseline_window))
            baseline_std = float(np.nanstd(baseline_window))

            if baseline_std == 0:
                baseline_std = 1.0

            cusum_pos = 0.0
            cusum_neg = 0.0
            current_baseline = baseline_mean

            for i in range(self.baseline_samples, n):
                val = rtt[i]
                if np.isnan(val):
                    continue

                z = (val - current_baseline) / baseline_std
                cusum_pos = max(0.0, cusum_pos + z - self.drift)
                cusum_neg = max(0.0, cusum_neg - z - self.drift)

                if cusum_pos > self.sensitivity or cusum_neg > self.sensitivity:
                    change_pct = (
                        (val - current_baseline) / current_baseline * 100
                        if current_baseline > 0
                        else 0.0
                    )
                    results.append({
                        "timestamp": timestamps[i],
                        "destination": host,
                        "new_rtt": float(val),
                        "baseline_rtt": float(current_baseline),
                        "rtt_change_pct": float(change_pct),
                    })

                    # Reset CUSUM and update baseline
                    lookback = max(0, i - self.baseline_samples)
                    recent = rtt[lookback: i + 1]
                    current_baseline = float(np.nanmean(recent))
                    cusum_pos = 0.0
                    cusum_neg = 0.0

        if not results:
            return pd.DataFrame(columns=[
                "timestamp", "destination", "new_rtt", "baseline_rtt",
                "rtt_change_pct",
            ])

        return pd.DataFrame(results)


class JitterDiagnosisMatrix:
    """Root cause classification using jitter, RTT, and packet loss patterns.

    Maps concurrent symptom combinations to likely network conditions,
    producing a diagnosis and severity level for each measurement.
    """

    # Diagnosis constants
    INFRASTRUCTURE_FAILURE = "INFRASTRUCTURE_FAILURE"
    CONGESTION = "CONGESTION"
    PATH_CHANGE_OR_ROUTING = "PATH_CHANGE_OR_ROUTING"
    LINK_SATURATION = "LINK_SATURATION"
    NORMAL = "NORMAL"

    def __init__(self, percentile: float = 0.95) -> None:
        self.percentile = percentile
        self.thresholds: dict[str, dict[str, float]] = {}

    def train(self, data: pd.DataFrame) -> None:
        """Learn per-destination p95 thresholds for rtt, jitter, and loss.

        Uses the fraction given by ``self.percentile`` (default 0.95).
        """
        q = self.percentile * 100  # numpy percentile uses 0-100

        for host, group in data.groupby("target_host"):
            rtt = group["rtt_avg"].dropna().values
            jitter_vals = group["jitter"].dropna().values
            sent = group["pkts_sent"].fillna(0).values
            lost = group["pkts_lost"].fillna(0).values

            loss_pct = np.where(sent > 0, lost / sent * 100, 0.0)

            self.thresholds[host] = {
                "rtt": float(np.percentile(rtt, q)) if len(rtt) > 0 else 0.0,
                "jitter": float(np.percentile(jitter_vals, q)) if len(jitter_vals) > 0 else 0.0,
                "packet_loss": float(np.percentile(loss_pct, q)) if len(loss_pct) > 0 else 0.0,
            }

    def diagnose(self, data: pd.DataFrame) -> pd.DataFrame:
        """Classify each measurement into a diagnosis category.

        Diagnosis matrix
        ~~~~~~~~~~~~~~~~
        High RTT + High Jitter + High Loss  -> INFRASTRUCTURE_FAILURE (CRITICAL)
        High Jitter + Normal RTT             -> CONGESTION (MEDIUM)
        High RTT + Normal Jitter             -> PATH_CHANGE_OR_ROUTING (HIGH)
        High RTT + High Jitter               -> LINK_SATURATION (HIGH)
        Otherwise                            -> NORMAL

        Returns
        -------
        pd.DataFrame
            Columns: timestamp, destination, diagnosis, severity, rtt, jitter,
            packet_loss.  Only non-NORMAL rows are returned.
        """
        rows: list[dict] = []

        for idx in range(len(data)):
            row = data.iloc[idx]
            host = row.get("target_host")
            thresh = self.thresholds.get(host)

            if thresh is None:
                continue

            rtt = row.get("rtt_avg")
            jitter_val = row.get("jitter")
            sent = row.get("pkts_sent")
            lost = row.get("pkts_lost")

            if rtt is None or np.isnan(rtt):
                continue

            jitter_val = jitter_val if (jitter_val is not None and not np.isnan(jitter_val)) else 0.0
            loss_pct = (lost / sent * 100) if (sent and sent > 0) else 0.0

            high_rtt = rtt > thresh["rtt"]
            high_jitter = jitter_val > thresh["jitter"]
            high_loss = loss_pct > thresh["packet_loss"]

            if high_rtt and high_jitter and high_loss:
                diagnosis = self.INFRASTRUCTURE_FAILURE
                severity = "CRITICAL"
            elif high_rtt and high_jitter:
                diagnosis = self.LINK_SATURATION
                severity = "HIGH"
            elif high_rtt and not high_jitter:
                diagnosis = self.PATH_CHANGE_OR_ROUTING
                severity = "HIGH"
            elif high_jitter and not high_rtt:
                diagnosis = self.CONGESTION
                severity = "MEDIUM"
            else:
                continue  # NORMAL -- skip

            rows.append({
                "timestamp": row.get("timestamp"),
                "destination": host,
                "diagnosis": diagnosis,
                "severity": severity,
                "rtt": float(rtt),
                "jitter": float(jitter_val),
                "packet_loss": float(loss_pct),
            })

        if not rows:
            return pd.DataFrame(columns=[
                "timestamp", "destination", "diagnosis", "severity",
                "rtt", "jitter", "packet_loss",
            ])

        return pd.DataFrame(rows)


class OutageClassifier:
    """Decision-tree classifier for network outage types.

    Combines ping metrics with optional traceroute hop data and DNS
    resolution status to produce a categorical classification, a
    confidence score, and a human-readable detail string.
    """

    # Classification labels
    HEALTHY = "HEALTHY"
    DNS_FAILURE = "DNS_FAILURE"
    LOCAL_FIRST_MILE_OUTAGE = "LOCAL_FIRST_MILE_OUTAGE"
    ROUTING_BLACKHOLE = "ROUTING_BLACKHOLE"
    ROUTING_LOOP = "ROUTING_LOOP"
    CONGESTION_AT_HOP = "CONGESTION_AT_HOP"
    HIGH_LATENCY = "HIGH_LATENCY"
    MILD_DEGRADATION = "MILD_DEGRADATION"
    PARTIAL_PACKET_LOSS = "PARTIAL_PACKET_LOSS"

    def classify(
        self,
        ping_data: dict,
        traceroute_hops: list[dict] | None = None,
        dns_data: dict | None = None,
        baseline_rtt: float = 50,
    ) -> tuple[str, float, str]:
        """Classify a single measurement into an outage category.

        Parameters
        ----------
        ping_data : dict
            Must contain keys ``rtt_avg``, ``pkts_sent``, ``pkts_received``,
            ``pkts_lost``.
        traceroute_hops : list[dict] | None
            Optional list of hop dicts, each with ``hop_number``,
            ``hop_ip``, ``hop_rtt``.
        dns_data : dict | None
            Optional dict with ``status`` (e.g. ``'ok'``, ``'fail'``).
        baseline_rtt : float
            Expected healthy RTT for comparison.

        Returns
        -------
        tuple[str, float, str]
            ``(classification, confidence, details)``
        """
        sent = ping_data.get("pkts_sent", 0) or 0
        received = ping_data.get("pkts_received", 0) or 0
        lost = ping_data.get("pkts_lost", 0) or 0
        rtt = ping_data.get("rtt_avg")

        loss_pct = (lost / sent * 100) if sent > 0 else 0.0
        ping_ok = received > 0

        # --- DNS failure with reachable host --------------------------------
        dns_fail = (
            dns_data is not None
            and str(dns_data.get("status", "")).lower() not in ("ok", "noerror", "")
        )
        if dns_fail and ping_ok:
            return (
                self.DNS_FAILURE,
                0.85,
                "DNS resolution failed but host responds to ping; likely DNS issue.",
            )

        # --- Total packet loss (100%) ----------------------------------------
        if loss_pct >= 100:
            if traceroute_hops is not None:
                max_hop = max(
                    (h.get("hop_number", 0) for h in traceroute_hops), default=0
                )
                if max_hop <= 2:
                    return (
                        self.LOCAL_FIRST_MILE_OUTAGE,
                        0.90,
                        f"Total packet loss with traceroute reaching only {max_hop} hop(s); "
                        "first-mile / local network failure.",
                    )

                # Check for routing loop: same IP appears at multiple hops
                if self._has_routing_loop(traceroute_hops):
                    return (
                        self.ROUTING_LOOP,
                        0.80,
                        "Total packet loss with routing loop detected in traceroute.",
                    )

                # Destination not reached
                target_reached = any(
                    h.get("hop_ip") == ping_data.get("target_host")
                    for h in traceroute_hops
                )
                if not target_reached:
                    return (
                        self.ROUTING_BLACKHOLE,
                        0.75,
                        "Total packet loss; traceroute does not reach destination.",
                    )

            # No traceroute data, just total loss
            return (
                self.ROUTING_BLACKHOLE,
                0.60,
                "Total packet loss with no traceroute data available.",
            )

        # --- Partial packet loss (0 < loss < 100) ----------------------------
        if 0 < loss_pct < 100:
            # Check for routing loop even with partial loss
            if traceroute_hops and self._has_routing_loop(traceroute_hops):
                return (
                    self.ROUTING_LOOP,
                    0.75,
                    f"Partial packet loss ({loss_pct:.1f}%) with routing loop detected.",
                )

            # Check for congestion at a specific hop
            congestion_hop = self._find_congestion_hop(traceroute_hops)
            if congestion_hop is not None:
                return (
                    self.CONGESTION_AT_HOP,
                    0.70,
                    f"Partial packet loss ({loss_pct:.1f}%) with latency spike at "
                    f"hop {congestion_hop['hop_number']} ({congestion_hop['hop_ip']}).",
                )

            return (
                self.PARTIAL_PACKET_LOSS,
                0.65,
                f"Partial packet loss ({loss_pct:.1f}%) without clear congestion point.",
            )

        # --- Latency-based classification ------------------------------------
        if rtt is not None and not np.isnan(rtt) and baseline_rtt > 0:
            ratio = rtt / baseline_rtt

            if ratio > 2.0:
                return (
                    self.HIGH_LATENCY,
                    0.70,
                    f"RTT ({rtt:.1f} ms) is {ratio:.1f}x the baseline ({baseline_rtt:.1f} ms).",
                )
            if ratio > 1.5:
                return (
                    self.MILD_DEGRADATION,
                    0.55,
                    f"RTT ({rtt:.1f} ms) is {ratio:.1f}x the baseline ({baseline_rtt:.1f} ms).",
                )

        return (
            self.HEALTHY,
            0.90,
            "All metrics within normal parameters.",
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _has_routing_loop(hops: list[dict]) -> bool:
        """Return True if any IP appears at more than one hop number."""
        seen: dict[str, int] = {}
        for h in hops:
            ip = h.get("hop_ip")
            hop_num = h.get("hop_number")
            if ip is None or hop_num is None:
                continue
            if ip in seen and seen[ip] != hop_num:
                return True
            seen[ip] = hop_num
        return False

    @staticmethod
    def _find_congestion_hop(hops: list[dict] | None) -> dict | None:
        """Return the hop dict showing the largest RTT jump, if any.

        A hop is considered a congestion point when its RTT is more than
        double the previous hop's RTT and the absolute increase exceeds
        50 ms.
        """
        if not hops:
            return None

        sorted_hops = sorted(hops, key=lambda h: h.get("hop_number", 0))
        worst: dict | None = None
        worst_increase = 0.0

        for i in range(1, len(sorted_hops)):
            prev_rtt = sorted_hops[i - 1].get("hop_rtt")
            curr_rtt = sorted_hops[i].get("hop_rtt")

            if prev_rtt is None or curr_rtt is None:
                continue
            if not isinstance(prev_rtt, (int, float)) or not isinstance(curr_rtt, (int, float)):
                continue

            increase = curr_rtt - prev_rtt
            if increase > 50 and curr_rtt > 2 * prev_rtt and increase > worst_increase:
                worst = sorted_hops[i]
                worst_increase = increase

        return worst
