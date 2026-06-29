# Inventor Network Analysis Toolkit

## Overview

`inventor_analysis` is a Python package for analyzing network performance monitoring data from the [Inventor](https://inventor.2lab.net) active measurement system. It ingests JSONL measurement files (ping, DNS, traceroute), runs exploratory analysis, detects anomalies with adaptive baselines, and classifies root causes of network degradation. Only numpy, pandas, matplotlib, seaborn, and scipy are required — no sklearn.

---

## Architecture

Data flows through three layers: **loading** (JSONL parsing into DataFrames), **analysis** (per-protocol EDA and statistical profiling), and **detection** (anomaly detection, root cause classification, outage typing). Visualization functions operate on outputs from any layer.

```
JSONL files ──► loaders ──► ping_analysis   ──► visualization
                            dns_analysis        (plots, dashboards)
                            traceroute_analysis
                                │
                                ▼
                          anomaly_detection
                          (baseline, CUSUM, jitter matrix, outage classifier)
```

For the full description of classes, evaluation logic, and design decisions see [docs/developer.md](docs/developer.md).

---

## Why

Network monitoring tools show raw metrics — RTT, packet loss, jitter. This toolkit turns those numbers into answers:

| Raw metric | What this toolkit tells you |
|---|---|
| RTT = 142ms | Whether 142ms is *abnormal* for this host at this hour |
| Latency spike | Whether it's congestion, routing change, link saturation, or hardware failure |
| 15-hop traceroute | Which specific hop introduced the degradation |
| Three DNS resolvers | Which one is fastest, most stable, and most consistent — with data |

**Use cases.** Daily health reports over 14-day windows. Incident investigation (alert fires → Jitter Diagnosis Matrix classifies the problem type → root cause localization pinpoints the hop). Resolver migration decisions backed by statistical comparison. Routing change impact assessment with CUSUM shift detection.

**Audience.** Network operators, NOC/SOC teams, researchers studying routing or DNS behavior, students working with real-world network data.

---

## Modules

### Data Loaders

`inventor_analysis.loaders`

Parses Inventor JSONL files (one JSON object per line, daily files) into analysis-ready DataFrames. Handles null `Result` records, empty summaries, non-responding traceroute hops, and inconsistent field naming (`pkts_send` vs `pkts_sent`).

```python
from inventor_analysis.loaders import load_ping_data, load_dns_data, load_traceroute_data

ping_df = load_ping_data("data/ping.internet", max_files=14)
dns_df = load_dns_data("data/dns.internet", max_files=14)
trace_summary, trace_hops = load_traceroute_data("data/traceroute.internet")
```

### Ping Analysis

`inventor_analysis.ping_analysis`

Per-host RTT aggregation, trend analysis, Z-score anomaly detection, and reliability scoring.

| Function | Purpose |
|---|---|
| `compute_host_statistics(df)` | Per-host: mean, std, median, p95, p99 RTT + jitter + loss |
| `compute_daily_trends(df)` | Daily RTT, jitter, packet loss aggregation |
| `compute_hourly_patterns(df)` | Hour-of-day patterns (diurnal cycles) |
| `detect_anomalies_zscore(df, threshold)` | Per-host Z-score anomaly flagging |
| `compute_loss_rate(df)` | Adds `loss_rate` column |
| `rank_hosts(df, metric, top_n)` | Host ranking by any metric |
| `compute_reliability_scores(df)` | Success rate, SLA breach %, loss % per host |

```python
from inventor_analysis.ping_analysis import compute_host_statistics, detect_anomalies_zscore

stats = compute_host_statistics(ping_df)
anomalies = detect_anomalies_zscore(ping_df, threshold=3.0)
```

### DNS Analysis

`inventor_analysis.dns_analysis`

Resolver comparison: speed, stability (coefficient of variation), IP set consistency (Jaccard similarity), TTL-response correlation, and combined ranking.

| Function | Purpose |
|---|---|
| `compute_resolver_statistics(df)` | Per-resolver response time stats + success rate |
| `compute_resolver_stability(df)` | Coefficient of variation per resolver |
| `detect_dns_anomalies(df, threshold)` | Per-resolver Z-score anomaly detection |
| `compute_threshold_breaches(df, threshold_ms)` | Fraction of queries exceeding latency threshold |
| `compute_resolver_consistency(df)` | IP set stability across days (Jaccard similarity) |
| `compute_ttl_response_correlation(df)` | TTL vs. response time correlation (Pearson) |
| `rank_resolvers(df)` | Combined ranking: speed + stability + reliability |

```python
from inventor_analysis.dns_analysis import rank_resolvers

rankings = rank_resolvers(dns_df)
```

### Traceroute Analysis

`inventor_analysis.traceroute_analysis`

Hop-level latency decomposition, gateway profiling, path fingerprinting, and root cause localization.

| Function | Purpose |
|---|---|
| `compute_hop_statistics(hops_df)` | Per-hop-position RTT statistics |
| `compute_gateway_performance(hops_df)` | Per-gateway-IP performance profile |
| `compute_incremental_latency(hops_df)` | Hop-to-hop latency deltas |
| `get_path_signatures(hops_df)` | Path fingerprinting (IP sequence per destination) |
| `compute_path_stability(summaries_df)` | Per-destination route stability |
| `build_hop_baselines(hops_df)` | Per-hop baseline stats for anomaly comparison |
| `localize_root_cause(hops, baselines)` | Identify the hop that caused degradation |
| `get_network_segments(hops_df, n_hops)` | Extract shared infrastructure segments |

```python
from inventor_analysis.traceroute_analysis import build_hop_baselines, localize_root_cause

baselines = build_hop_baselines(trace_hops)
result = localize_root_cause(current_hops, baselines)
# "Hop 3 (147.229.253.233) is 49.6ms above baseline (15.2 std)"
```

### Anomaly Detection

`inventor_analysis.anomaly_detection`

Five models for detection, shift analysis, and root cause classification:

| Class | Purpose |
|---|---|
| `AdaptiveBaselineDetector` | p99.9 thresholds + multi-signal verification (precision >0.6) |
| `CorrelatedDegradationDetector` | Network-wide incident detection (multiple destinations degrade simultaneously) |
| `LatencyShiftDetector` | CUSUM algorithm for sustained latency level changes |
| `JitterDiagnosisMatrix` | Root cause classification: congestion / routing / saturation / failure |
| `OutageClassifier` | Decision-tree outage typing from ping + traceroute + DNS signals |

```python
from inventor_analysis.anomaly_detection import AdaptiveBaselineDetector, JitterDiagnosisMatrix

detector = AdaptiveBaselineDetector()
detector.train(train_data)
predictions, scores, reasons = detector.predict(test_data)

diagnoser = JitterDiagnosisMatrix(percentile=0.95)
diagnoser.train(train_data)
diagnoses = diagnoser.diagnose(test_data)
```

For full API reference, constructor parameters, return formats, and diagnosis matrix see [docs/usage.md](docs/usage.md).

### Visualization

`inventor_analysis.visualization`

12 plotting functions with sensible defaults. All accept an optional `ax` parameter for embedding in custom layouts.

```python
from inventor_analysis.visualization import create_summary_dashboard

fig = create_summary_dashboard(ping_df, dns_df, trace_summary)
fig.savefig('dashboard.png', dpi=150)
```

For the full function list see [docs/usage.md](docs/usage.md).

---

## Quick Start

```python
from inventor_analysis.loaders import load_ping_data
from inventor_analysis.ping_analysis import compute_host_statistics
from inventor_analysis.anomaly_detection import AdaptiveBaselineDetector, JitterDiagnosisMatrix

df = load_ping_data("data/ping.internet", max_files=14)
stats = compute_host_statistics(df)

detector = AdaptiveBaselineDetector()
detector.train(df)
predictions, scores, reasons = detector.predict(df)

diagnoser = JitterDiagnosisMatrix(percentile=0.95)
diagnoser.train(df)
diagnoses = diagnoser.diagnose(df)
print(diagnoses['diagnosis'].value_counts())
```

For installation, dependencies, and deployment see [docs/deployment.md](docs/deployment.md). For data format schemas and demo walkthrough see [docs/usage.md](docs/usage.md).

---

## Package Structure

```
final/
├── inventor_analysis/
│   ├── __init__.py
│   ├── loaders.py               # JSONL data loading
│   ├── ping_analysis.py         # Ping EDA
│   ├── dns_analysis.py          # DNS resolver comparison
│   ├── traceroute_analysis.py   # Hop-level analysis
│   ├── anomaly_detection.py     # Anomaly detection + diagnosis
│   └── visualization.py         # Plotting functions
├── examples/                    # Demo notebooks
├── sample_data/                 # Sample JSONL extracts
├── docs/
│   ├── usage.md                 # Full API reference, data format, demos
│   ├── developer.md             # Internals, design decisions, dead ends
│   └── deployment.md            # Installation, requirements, integration
├── requirements.txt
└── README.md
```

---

## License

This software is provided as part of a network analysis research project.
