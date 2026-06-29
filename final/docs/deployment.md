# Deployment

## Requirements

- Python 3.9+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`

No build step. No package installation. Copy the `final/` directory and install dependencies.

---

## Setup

```bash
cd final/
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### Dependencies

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
```

### Verify

```python
from inventor_analysis import load_ping_data, load_dns_data, load_traceroute_data
```

---

## File Layout

```
final/
├── inventor_analysis/           # Main package — add this directory's parent to PYTHONPATH
│   ├── __init__.py
│   ├── loaders.py
│   ├── ping_analysis.py
│   ├── dns_analysis.py
│   ├── traceroute_analysis.py
│   ├── anomaly_detection.py
│   └── visualization.py
├── examples/                    # Demo notebooks
│   ├── demo_ping_analysis.ipynb
│   ├── demo_dns_analysis.ipynb
│   ├── demo_anomaly_detection.ipynb
│   └── demo_full_pipeline.ipynb
├── sample_data/                 # 20-line extracts from real monitoring data
│   ├── ping_sample.json
│   ├── dns_sample.json
│   └── traceroute_sample.json
├── requirements.txt
└── README.md
```

No package installation is required. The `inventor_analysis/` directory is a standard Python package importable from its parent directory. The demo notebooks handle this by inserting `..` into `sys.path`.

---

## Integration Patterns

### Script usage

Import directly in any Python script. Ensure the `final/` directory is on `PYTHONPATH` or use `sys.path.insert`.

```python
import sys
sys.path.insert(0, '/path/to/final')

from inventor_analysis.loaders import load_ping_data
from inventor_analysis.anomaly_detection import AdaptiveBaselineDetector
```

### Batch / cron

Run custom analysis scripts on a schedule against rotating daily data directories, or use `jupyter nbconvert --execute` to run the demo notebooks headlessly.

```bash
cd /path/to/final/examples
jupyter nbconvert --execute --to notebook demo_full_pipeline.ipynb
```

### Data directory layout

Point the loaders at a directory containing Inventor JSONL files. The loaders scan for `*.json` files, sort by filename (which embeds the date), and parse each file line by line.

```python
df = load_ping_data("/var/data/inventor/ping.internet", max_files=14)
```

The `max_files` parameter limits how many recent files to load (sorted by filename descending). Omit it to load all files in the directory.

---

## Visualization Backend

The visualization module uses matplotlib. For headless environments (servers, CI), set the backend before importing:

```python
import matplotlib
matplotlib.use('Agg')

from inventor_analysis.visualization import create_summary_dashboard
```

All plotting functions return matplotlib `Axes` or `Figure` objects for saving to file.
