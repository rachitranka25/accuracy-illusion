"""Nifty Oracle — intraday research platform for NSE index derivatives.

Layout:
    oracle.data      broker + exchange data collection
    oracle.features  indicators and positioning features
    oracle.models    direction, volatility and meta-label models
    oracle.strategy  option structures and positioning signals
    oracle.app       the live dashboard server
"""
from . import paths  # noqa: F401

__version__ = "1.0.0"
