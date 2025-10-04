"""
Financial Market Data Toolkit

A Python 3 toolkit for fetching, cleaning, and analyzing financial market data.
Includes modules for multi-API extraction, standardized price dataclasses,
portfolio simulation (Monte Carlo), markdown reporting, and visualization.
"""

from .data_models import PriceData, Portfolio
from .data_extractor import DataExtractor
from .data_cleaner import DataCleaner
from .portfolio_simulator import PortfolioSimulator
from .report_generator import ReportGenerator
from .visualizer import Visualizer

__version__ = "1.0.0"
__all__ = [
    "PriceData",
    "Portfolio",
    "DataExtractor",
    "DataCleaner",
    "PortfolioSimulator",
    "ReportGenerator",
    "Visualizer",
]
