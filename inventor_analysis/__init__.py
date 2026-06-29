"""
Inventor Network Analysis Toolkit

A comprehensive Python package for analyzing network performance monitoring data
from the Inventor active monitoring system. Supports ping (ICMP), DNS, and
traceroute data analysis with built-in anomaly detection and root cause diagnosis.
"""

__version__ = "1.0.0"

from .loaders import load_ping_data, load_dns_data, load_traceroute_data
