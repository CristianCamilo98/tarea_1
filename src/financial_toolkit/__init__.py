"""
Financial Market Data Toolkit

A Python 3 toolkit for fetching, cleaning, and analyzing financial market data.
Includes modules for multi-API extraction, standardized price dataclasses,
portfolio simulation (Monte Carlo), markdown reporting, and visualization.
"""

from .models.market_data import PriceData, Portfolio
from .data_extractor import DataExtractor

__version__ = "1.0.0"
__all__ = [
    "PriceData",
    "Portfolio",
    "DataExtractor",
]
