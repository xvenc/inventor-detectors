"""Data loading functions for the Inventor network monitoring system.

Loads JSONL files (one JSON object per line) organized by date into
pandas DataFrames suitable for analysis.
"""

from __future__ import annotations

import json
import re
from glob import glob
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd


_DATE_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})")


def _extract_file_date(path: str | Path) -> str | None:
    match = _DATE_PATTERN.search(Path(path).name)
    return match.group(1) if match else None


def _iter_json_lines(file_path: str | Path) -> Iterator[dict]:
    with open(file_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _sorted_files(directory: str | Path, pattern: str, max_files: int | None) -> list[Path]:
    dir_path = Path(directory)
    files = list(dir_path.glob(pattern))
    if not files:
        files = list(dir_path.glob("*.json"))
    files.sort(key=lambda p: _extract_file_date(p) or "")
    if max_files is not None:
        files = files[:max_files]
    return files


def load_ping_data(directory: str | Path, max_files: int | None = None) -> pd.DataFrame:
    files = _sorted_files(directory, "network.ping.*.json", max_files)
    rows: list[dict] = []

    for file_path in files:
        file_date = _extract_file_date(file_path)
        for record in _iter_json_lines(file_path):
            result = record.get("Result")
            if result is None:
                continue
            summary = result.get("summary")
            if not summary:
                continue

            meta = record.get("Meta", {})
            config = record.get("Config", {})

            rows.append({
                "timestamp": meta.get("Timestamp"),
                "test_id": meta.get("TestId"),
                "target_host": config.get("target_host"),
                "packet_size": config.get("packet_size"),
                "packet_count": config.get("packet_count"),
                "pkts_sent": summary.get("pkts_send"),
                "pkts_received": summary.get("pkts_received"),
                "pkts_lost": summary.get("pkts_lost"),
                "rtt_min": summary.get("rtt_min"),
                "rtt_max": summary.get("rtt_max"),
                "rtt_avg": summary.get("rtt_avg"),
                "rtt_stddev": summary.get("rtt_stddev"),
                "jitter": summary.get("jitter"),
                "file_date": file_date,
            })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "test_id", "target_host", "packet_size", "packet_count",
            "pkts_sent", "pkts_received", "pkts_lost", "rtt_min", "rtt_max",
            "rtt_avg", "rtt_stddev", "jitter", "file_date",
        ])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce").dt.date

    numeric_cols = [
        "packet_size", "packet_count", "pkts_sent", "pkts_received",
        "pkts_lost", "rtt_min", "rtt_max", "rtt_avg", "rtt_stddev", "jitter",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_dns_data(directory: str | Path, max_files: int | None = None) -> pd.DataFrame:
    files = _sorted_files(directory, "network.dns.*.json", max_files)
    rows: list[dict] = []

    for file_path in files:
        file_date = _extract_file_date(file_path)
        for record in _iter_json_lines(file_path):
            result = record.get("Result")
            if result is None:
                continue

            meta = record.get("Meta", {})
            config = record.get("Config", {})
            details = result.get("details", [])
            if not details:
                continue

            nameservers = config.get("nameservers", [])
            resolver = nameservers[0] if nameservers else None

            for detail in details:
                ips = detail.get("IP_address", [])

                rows.append({
                    "timestamp": meta.get("Timestamp"),
                    "test_id": meta.get("TestId"),
                    "resolver": resolver,
                    "query_type": config.get("query_type"),
                    "target_host": detail.get("target_host"),
                    "response_time_ms": detail.get("response_time"),
                    "ttl_s": detail.get("expiration_time"),
                    "status": detail.get("status"),
                    "status_code": detail.get("status_code"),
                    "ip_count": len(ips) if isinstance(ips, list) else 0,
                    "ips": ips,
                    "file_date": file_date,
                })

    if not rows:
        return pd.DataFrame(columns=[
            "timestamp", "test_id", "resolver", "query_type", "target_host",
            "response_time_ms", "ttl_s", "status", "status_code",
            "ip_count", "ips", "file_date",
        ])

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["file_date"] = pd.to_datetime(df["file_date"], errors="coerce").dt.date
    df["response_time_ms"] = pd.to_numeric(df["response_time_ms"], errors="coerce")
    df["ttl_s"] = pd.to_numeric(df["ttl_s"], errors="coerce")
    df["ip_count"] = pd.to_numeric(df["ip_count"], errors="coerce").astype(int)

    return df


def load_traceroute_data(
    directory: str | Path, max_files: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    files = _sorted_files(directory, "network.traceroute.*.json", max_files)
    summary_rows: list[dict] = []
    hop_rows: list[dict] = []

    for file_path in files:
        file_date = _extract_file_date(file_path)
        for record in _iter_json_lines(file_path):
            result = record.get("Result")
            if result is None:
                continue

            meta = record.get("Meta", {})
            config = record.get("Config", {})
            summary = result.get("summary")

            if summary:
                summary_rows.append({
                    "timestamp": meta.get("Timestamp"),
                    "test_id": meta.get("TestId"),
                    "target_host": config.get("target_host"),
                    "ip_address": summary.get("IP_address"),
                    "min_hops": summary.get("min_hops"),
                    "max_hops": summary.get("max_hops"),
                    "path_stability": summary.get("path_stability"),
                    "packet_loss": summary.get("packet_loss"),
                    "file_date": file_date,
                })

            for run_detail in result.get("details", []):
                run_id = run_detail.get("run")
                for hop in run_detail.get("hops", []):
                    # Non-responding gateways have a string instead of numeric RTT
                    if not isinstance(hop.get("hop_rtt"), (int, float)):
                        continue
                    hop_rows.append({
                        "timestamp": meta.get("Timestamp"),
                        "target_host": config.get("target_host"),
                        "run_id": run_id,
                        "hop_number": hop.get("hop_number"),
                        "hop_ip": hop.get("hop_ip"),
                        "hop_rtt": hop.get("hop_rtt"),
                    })

    summary_cols = [
        "timestamp", "test_id", "target_host", "ip_address",
        "min_hops", "max_hops", "path_stability", "packet_loss", "file_date",
    ]
    hop_cols = [
        "timestamp", "target_host", "run_id", "hop_number", "hop_ip", "hop_rtt",
    ]

    if not summary_rows:
        summaries_df = pd.DataFrame(columns=summary_cols)
    else:
        summaries_df = pd.DataFrame(summary_rows)
        summaries_df["timestamp"] = pd.to_datetime(summaries_df["timestamp"], errors="coerce")
        summaries_df["file_date"] = pd.to_datetime(summaries_df["file_date"], errors="coerce").dt.date
        for col in ["min_hops", "max_hops", "path_stability", "packet_loss"]:
            summaries_df[col] = pd.to_numeric(summaries_df[col], errors="coerce")

    if not hop_rows:
        hops_df = pd.DataFrame(columns=hop_cols)
    else:
        hops_df = pd.DataFrame(hop_rows)
        hops_df["timestamp"] = pd.to_datetime(hops_df["timestamp"], errors="coerce")
        hops_df["hop_number"] = pd.to_numeric(hops_df["hop_number"], errors="coerce")
        hops_df["hop_rtt"] = pd.to_numeric(hops_df["hop_rtt"], errors="coerce")
        hops_df["run_id"] = pd.to_numeric(hops_df["run_id"], errors="coerce")

    return summaries_df, hops_df
