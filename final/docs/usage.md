# Usage

## Data Format

Inventor stores measurements as **JSONL** (one JSON object per line), one file per day. Each record has `Meta` (timestamp, test ID), `Config` (test parameters), and `Result` (measurements).

---

### Ping

Filename: `network.ping.internet.YYYY-MM-DD.json`

```json
{
  "Meta": {"Timestamp": "2025-05-16T10:25:40", "TestId": "network.ping.60"},
  "Config": {"target_host": "google.com", "packet_size": 100, "packet_count": 3},
  "Result": {
    "status": "completed",
    "summary": {
      "pkts_send": 3, "pkts_received": 3, "pkts_lost": 0.0,
      "rtt_min": 3.798, "rtt_max": 3.940, "rtt_avg": 3.887,
      "rtt_stddev": 0.078, "jitter": 0.071
    }
  }
}
```

**DataFrame columns:** `timestamp`, `test_id`, `target_host`, `rtt_avg`, `rtt_min`, `rtt_max`, `rtt_stddev`, `jitter`, `pkts_sent`, `pkts_received`, `pkts_lost`, `file_date`

---

### DNS

Filename: `network.dns.internet.YYYY-MM-DD.json`

```json
{
  "Meta": {"Timestamp": "2025-06-20T10:25:08", "TestId": "network.dns.7"},
  "Config": {"query_type": "A", "target_hosts": ["example.com"], "nameservers": ["9.9.9.9"]},
  "Result": {
    "status": "completed",
    "details": [{
      "target_host": "example.com.",
      "IP_address": ["23.215.0.138", "23.192.228.84"],
      "expiration_time": 299,
      "response_time": 99.71,
      "status": "success", "status_code": "NOERROR"
    }]
  }
}
```

**DataFrame columns:** `timestamp`, `resolver`, `query_type`, `target_host`, `response_time_ms`, `ttl_s`, `status`, `status_code`, `ip_count`, `ips`, `file_date`

---

### Traceroute

Filename: `network.traceroute.YYYY-MM-DD.json`

```json
{
  "Meta": {"Timestamp": "2025-09-26T15:11:34", "TestId": "network.traceroute.4"},
  "Config": {"target_host": "8.8.8.8", "ttl_max": 30},
  "Result": {
    "status": "completed",
    "summary": {"min_hops": 9, "max_hops": 9, "path_stability": 1.0, "packet_loss": 0.0},
    "details": [{"run": 1, "hops": [
      {"hop_number": 1, "hop_ip": "147.229.14.1", "hop_rtt": 1.547},
      {"hop_number": 2, "hop_ip": "147.229.254.68", "hop_rtt": 0.306},
      {"hop_number": 9, "hop_ip": "8.8.8.8", "hop_rtt": 4.212}
    ]}]
  }
}
```

**Returns two DataFrames:**
- Summaries: `timestamp`, `target_host`, `min_hops`, `max_hops`, `path_stability`, `packet_loss`
- Hops: `timestamp`, `target_host`, `hop_number`, `hop_ip`, `hop_rtt`

---

## Anomaly Detection API

### AdaptiveBaselineDetector

Ultra-strict anomaly detector. p99.9 thresholds + multi-signal verification. Designed for precision >0.6 in production.

**Detection logic.** RTT must exceed p99.9 AND absolute threshold (>100ms). Requires corroborating signal: high jitter (>p99), severe packet loss (>30%), or massive deviation (>10x MAD). Special case: catastrophic failure (>50% loss AND RTT > p95).

```python
from inventor_analysis.anomaly_detection import AdaptiveBaselineDetector

detector = AdaptiveBaselineDetector()
detector.train(train_data)
predictions, severity_scores, reasons = detector.predict(test_data)
# predictions: boolean array
# severity_scores: float array (0-1)
# reasons: 'rtt_jitter' | 'rtt_loss' | 'extreme_rtt' | 'catastrophic' | 'none'
```

---

### CorrelatedDegradationDetector

Detects network-wide incidents — multiple destinations degrading simultaneously within a time window.

| Parameter | Default | Description |
|---|---|---|
| `time_window_minutes` | `5` | Grouping window for concurrent anomalies |
| `min_affected` | `3` | Minimum destinations affected to declare an incident |

```python
from inventor_analysis.anomaly_detection import CorrelatedDegradationDetector

corr = CorrelatedDegradationDetector(time_window_minutes=5, min_affected=3)
incidents = corr.detect(data, baseline_detector)
# Returns DataFrame: window_start, window_end, affected_count, affected_destinations, severity
```

---

### LatencyShiftDetector

CUSUM (Cumulative Sum) algorithm for sustained latency level changes. A spike at 500ms might be a measurement artifact; a sustained shift from 5ms to 15ms indicates a real network change.

| Parameter | Default | Description |
|---|---|---|
| `sensitivity` | `5.0` | Threshold for CUSUM sum to declare a shift |
| `drift` | `1.0` | Drift parameter to prevent accumulation of normal variation |
| `baseline_samples` | `50` | Number of initial samples for baseline computation |

```python
from inventor_analysis.anomaly_detection import LatencyShiftDetector

shift = LatencyShiftDetector(sensitivity=5.0, drift=1.0, baseline_samples=50)
shifts = shift.detect(data)
# Returns DataFrame: timestamp, destination, new_rtt, baseline_rtt, rtt_change_pct
```

---

### JitterDiagnosisMatrix

Concurrent root cause classification using jitter, RTT, and packet loss relationships.

| Condition | Diagnosis | Severity |
|---|---|---|
| High jitter + normal RTT | `CONGESTION` | MEDIUM |
| Normal jitter + high RTT | `PATH_CHANGE_OR_ROUTING` | HIGH |
| High jitter + high RTT | `LINK_SATURATION` | HIGH |
| High jitter + high RTT + high loss | `INFRASTRUCTURE_FAILURE` | CRITICAL |

| Parameter | Default | Description |
|---|---|---|
| `percentile` | `0.95` | Threshold percentile for "high" classification |

```python
from inventor_analysis.anomaly_detection import JitterDiagnosisMatrix

diagnoser = JitterDiagnosisMatrix(percentile=0.95)
diagnoser.train(train_data)
diagnoses = diagnoser.diagnose(test_data)
# Returns DataFrame with columns: target_host, timestamp, diagnosis, severity,
#                                  rtt_avg, jitter, pkts_lost
```

---

### OutageClassifier

Decision-tree outage type classification from ping + traceroute + DNS signals. Domain-knowledge-driven, no training data required. Each classification returns `(type, confidence, explanation)`.

**Categories:** `HEALTHY`, `DNS_FAILURE`, `LOCAL_FIRST_MILE_OUTAGE`, `ROUTING_BLACKHOLE`, `ROUTING_LOOP`, `CONGESTION_AT_HOP`, `HIGH_LATENCY`, `MILD_DEGRADATION`, `PARTIAL_PACKET_LOSS`

```python
from inventor_analysis.anomaly_detection import OutageClassifier

classifier = OutageClassifier()
classification, confidence, details = classifier.classify(
    ping_data={'ping_success': True, 'packet_loss': 100, 'rtt_avg': 0},
    traceroute_hops=[{'hop_num': 1, 'ip': '10.0.0.1', 'rtt': 1.0}],
    baseline_rtt=50
)
# -> ('LOCAL_FIRST_MILE_OUTAGE', 0.9, 'Failure within first 2 hops')
```

---

## Visualization Functions

All functions accept an optional `ax` parameter for embedding in custom matplotlib layouts.

| Function | Description |
|---|---|
| `plot_rtt_distribution(df, source_label)` | RTT histogram per host |
| `plot_host_comparison(df, metric, top_n)` | Bar chart comparison across hosts |
| `plot_daily_trend(daily_df, metric, label)` | Time series of daily metrics |
| `plot_hourly_pattern(hourly_df, metric)` | Diurnal cycle visualization |
| `plot_rtt_heatmap(df)` | Host × hour-of-day heatmap |
| `plot_anomaly_timeline(df)` | Anomaly scatter on timeline |
| `plot_dns_resolver_comparison(df)` | Resolver box plots |
| `plot_hop_latency_buildup(hops_df)` | Traceroute hop-by-hop RTT buildup |
| `plot_diagnosis_distribution(diagnoses_df)` | Diagnosis type bar chart |
| `create_summary_dashboard(ping_df, dns_df, trace_df)` | 3×3 combined dashboard |

```python
from inventor_analysis.visualization import plot_rtt_distribution, create_summary_dashboard

# Single plot
ax = plot_rtt_distribution(ping_df, source_label='Internet')
ax.figure.savefig('rtt_dist.png')

# Full dashboard
fig = create_summary_dashboard(ping_df, dns_df, trace_summary)
fig.savefig('dashboard.png', dpi=150)
```

---

## Demonstration Cases

Four Jupyter notebooks in `examples/`. Each uses `sample_data/` by default (20-line extracts from real monitoring data). Replace the data path with a full directory for production analysis.

```bash
cd examples/
jupyter notebook
```

| Notebook | Focus |
|---|---|
| `demo_ping_analysis.ipynb` | Host stats, trends, Z-score anomalies, visualizations |
| `demo_dns_analysis.ipynb` | Resolver comparison, stability, consistency |
| `demo_anomaly_detection.ipynb` | All five anomaly detection models |
| `demo_full_pipeline.ipynb` | End-to-end: load → analyze → detect → diagnose |

---

### demo_ping_analysis.ipynb

Loads ping JSONL data and produces per-host RTT statistics, fastest/slowest rankings, daily and hourly trends, Z-score anomaly detection, and reliability scores. Generates visualization PNGs.

**Sample output:**
```
HOST PERFORMANCE STATISTICS
              rtt_mean  rtt_std  rtt_p95  jitter_mean  packet_loss_mean  count
google.com       4.34     3.87    5.12         0.66              0.02   3832
facebook.com     4.52    15.82    4.89         0.91              0.02   3833
salesforce.com 187.95   147.25  276.81       254.95              0.00   3836

ANOMALY DETECTION (Z-Score > 3)
Total anomalies detected: 6,041 (0.86%)
```

---

### demo_dns_analysis.ipynb

Compares DNS resolver performance across speed, stability (coefficient of variation), consistency (Jaccard similarity), and anomaly rate.

**Sample output:**
```
RESOLVER PERFORMANCE STATISTICS
           rt_mean   rt_std  success_rate  query_count
8.8.8.8      4.47     0.39         1.000         3851
1.1.1.1      4.96     2.79         0.999         3851
9.9.9.9     21.12    32.46         1.000         3850
```

---

### demo_anomaly_detection.ipynb

Runs all five anomaly detection models on the same dataset: Adaptive Baseline, Correlated Degradation, CUSUM Latency Shifts, Jitter Diagnosis Matrix, and Outage Classifier with sample cases.

---

### demo_full_pipeline.ipynb

End-to-end pipeline combining all modules: load data from all three protocols → exploratory analysis → anomaly detection → latency shift detection → root cause diagnosis. Prints a summary of key findings.
