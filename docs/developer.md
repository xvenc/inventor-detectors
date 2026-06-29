# Developer Manual

## Project Structure

```
inventor_analysis/
  __init__.py                    # package init, version, top-level imports
  loaders.py                     # JSONL parsing for ping, DNS, traceroute
  ping_analysis.py               # per-host RTT statistics, trends, Z-score
  dns_analysis.py                # resolver comparison, stability, consistency
  traceroute_analysis.py         # hop-level analysis, root cause localization
  anomaly_detection.py           # 5 detection/classification models
  visualization.py               # 12 plotting functions
examples/
  demo_ping_analysis.ipynb
  demo_dns_analysis.ipynb
  demo_anomaly_detection.ipynb
  demo_full_pipeline.ipynb
sample_data/                     # 20-line extracts from real Inventor data
docs/
  usage.md
  deployment.md
  developer.md
```

The package is a flat module layout with no internal dependencies beyond the standard imports (numpy, pandas, scipy, matplotlib, seaborn). Each module can be imported independently.

---

## Data Classes and Models

| Class | Module | Purpose |
|---|---|---|
| `AdaptiveBaselineDetector` | `anomaly_detection` | Per-destination p99.9 baseline with multi-signal verification |
| `CorrelatedDegradationDetector` | `anomaly_detection` | Time-windowed multi-destination incident detection |
| `LatencyShiftDetector` | `anomaly_detection` | CUSUM-based sustained level change detection |
| `JitterDiagnosisMatrix` | `anomaly_detection` | Concurrent root cause classification from jitter/RTT/loss |
| `OutageClassifier` | `anomaly_detection` | Deterministic decision-tree outage typing |

All models follow a `train(df)` / `predict(df)` or `train(df)` / `diagnose(df)` pattern except `OutageClassifier`, which is stateless and classifies individual observations via `classify()`.

---

## Adaptive Baseline Detector

### Design rationale

Simple threshold alerting ("alert if RTT > 200ms") fails in two directions: a host at 140ms baseline never triggers, while a host at 4ms baseline misses a meaningful spike to 50ms. The adaptive baseline learns *what's normal for each destination* and flags deviations from that learned profile.

### Thresholds

Uses p99.9 instead of the more common p95/p99. This reduces false positives by only flagging the most extreme 0.1% of observations per destination.

### Multi-signal requirement

RTT anomaly alone is insufficient. The detector requires corroboration from at least one secondary signal:

| Reason code | RTT condition | Secondary condition |
|---|---|---|
| `rtt_jitter` | > p99.9 AND > 100ms | Jitter > p99 |
| `rtt_loss` | > p99.9 AND > 100ms | Packet loss > 30% |
| `extreme_rtt` | > p99.9 AND > 100ms | RTT deviation > 10× MAD |
| `catastrophic` | > p95 | Packet loss > 50% |

The absolute minimum threshold (100ms) prevents flagging statistically extreme but operationally irrelevant deviations on fast hosts (e.g., a CDN at 3ms spiking to 8ms is a 2.6× increase but not operationally meaningful).

### Precision target

The multi-signal approach was specifically designed to achieve precision >0.6. In production operations, a noisy detector gets ignored — operators stop trusting alerts after too many false positives. The p99.9 + corroboration design trades recall (misses some real anomalies) for precision (anomalies that are flagged are almost always real).

---

## Jitter Diagnosis Matrix

### Origin

Developed after discovering that a prior model's claim of "jitter predicts RTT 12 hours ahead" was a **seasonality artifact**. Both jitter and RTT follow the same diurnal traffic cycle, so any 12-hour shift aligns peak with peak — creating a spurious lagged correlation that disappears after detrending.

### Classification logic

Instead of prediction, jitter is used *concurrently* with RTT and packet loss for real-time root cause classification:

- **High jitter + normal RTT → CONGESTION.** Buffer contention without path change. Network is experiencing queue buildup at intermediate hops, increasing variability without increasing base latency. Typical at peering points during peak hours.

- **Normal jitter + high RTT → PATH_CHANGE_OR_ROUTING.** The path itself changed — a new, longer route is being used. Latency is consistently higher but stable (low jitter), indicating a routing change rather than load-related congestion.

- **High jitter + high RTT → LINK_SATURATION.** The physical link is at capacity. Both base latency and variability increase because packets queue and some experience significant delays. Common when a link is consistently near 100% utilization.

- **High jitter + high RTT + high loss → INFRASTRUCTURE_FAILURE.** Hardware or link is failing. All metrics degrade simultaneously — packets are being dropped, delayed, and reordered. Requires immediate attention.

### Thresholds

"High" is defined per-destination using the training data at a configurable percentile (default p95). This means a jitter value of 5ms is "high" for a host that normally has 0.5ms jitter, but "normal" for a host that routinely shows 10ms jitter.

---

## CUSUM Shift Detection

### Problem

Transient spike detectors (Z-score, threshold) fire on brief outliers but miss sustained level changes. A spike from 5ms to 500ms for one measurement is visible; a persistent shift from 5ms to 15ms is harder to detect but more operationally significant — it indicates an actual network change (routing, peering, congestion regime change).

### Algorithm

The CUSUM algorithm maintains a running cumulative sum of deviations from the learned baseline mean:

1. Compute baseline mean and standard deviation from the first `baseline_samples` measurements.
2. For each subsequent measurement, compute the deviation: `x - baseline_mean - drift`.
3. Add the deviation to the cumulative sum (clamped to ≥ 0).
4. When the sum exceeds `sensitivity × baseline_std`, declare a shift.
5. Reset the baseline to the current level and restart.

The `drift` parameter prevents accumulation of normal variation — small deviations below the drift are absorbed without contributing to the sum. The `sensitivity` parameter controls how large a shift must be before detection.

---

## Outage Classification

### Design choice

Uses a deterministic decision tree rather than ML classification because:

- Network outage types have well-understood, distinct signatures.
- A decision tree is interpretable — operators can understand *why* a classification was made.
- No training data is required — the logic is domain-knowledge-driven.
- Each classification includes a confidence score and human-readable explanation.

### Decision flow

The classifier combines three signal sources (ping, traceroute, DNS) and walks a priority-ordered decision tree:

1. **DNS check** — if DNS resolution failed, classify as `DNS_FAILURE`.
2. **Ping reachability** — if ping fails completely (100% loss), check traceroute for where the path dies.
3. **Traceroute analysis** — if hops terminate early, classify as `ROUTING_BLACKHOLE`. If hops loop, classify as `ROUTING_LOOP`. If a specific hop shows extreme latency, classify as `CONGESTION_AT_HOP`.
4. **RTT analysis** — if RTT > 5× baseline, classify as `HIGH_LATENCY`. If 2-5× baseline, classify as `MILD_DEGRADATION`.
5. **Loss analysis** — if partial packet loss without other symptoms, classify as `PARTIAL_PACKET_LOSS`.
6. **Default** — if no degradation signals, classify as `HEALTHY`.

---

## Data Loading Robustness

The loaders handle several edge cases from real-world Inventor monitoring:

| Edge case | Handling |
|---|---|
| `Result` is `null` | Record skipped (probe failed to execute) |
| `summary` is empty `{}` | Record skipped (probe ran but produced no summary) |
| Mixed JSONL and array-JSON | Both formats parsed transparently |
| Non-responding traceroute hops | Hop skipped (string value `"Some gateways are not responding"`) |
| `pkts_send` vs `pkts_sent` | Both field names accepted, normalized to `pkts_sent` |
| Missing numeric fields | Default to `NaN` rather than crashing |

### Date extraction

File dates are extracted from filenames matching the pattern `YYYY-MM-DD` (e.g., `network.ping.internet.2025-05-16.json` → `2025-05-16`). This is added as the `file_date` column, enabling time-based filtering and aggregation independent of the measurement timestamps.

---

## Dead Ends

### Jitter as a 12-hour RTT predictor

Early analysis showed strong lagged correlation between jitter and RTT shifted by 12 hours — jitter could seemingly predict future latency half a day ahead. After detrending both signals for daily traffic cycles, the correlation disappeared. Both metrics follow the same diurnal pattern; any 12-hour shift aligns peak with peak.

**Lesson.** Always detrend before claiming lagged correlations in time-series data with periodic components. Jitter was repurposed into the concurrent Jitter Diagnosis Matrix instead.

### Prophet RTT forecasting

Facebook Prophet model achieved R²=1.000 on validation. Investigation revealed data leakage: the feature set included the current RTT value, so the model learned the identity function. After removing leaked features and using proper walk-forward validation with only temporal features, predictive power was not useful for operational forecasting.

**Lesson.** A perfect R² on real-world data is almost always a bug. Check the feature matrix for leakage before celebrating.

### Impact on design

These failures informed the final design: the toolkit focuses on **detection and diagnosis** (where the data supports strong conclusions) rather than **prediction** (where the available signals proved insufficient for actionable forecasts).
