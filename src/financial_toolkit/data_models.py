"""
Data Models Module

Defines standardized dataclasses for financial market data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd


@dataclass
class PriceData:
    """
    Standardized price data structure for financial instruments.
    
    Attributes:
        symbol: Ticker symbol (e.g., 'AAPL', 'GOOGL')
        date: Date of the price data
        open: Opening price
        high: Highest price
        low: Lowest price
        close: Closing price
        volume: Trading volume
        adjusted_close: Adjusted closing price (optional)
        source: Data source identifier (e.g., 'yahoo', 'alpha_vantage')
    """
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    source: str = "unknown"
    
    def to_dict(self) -> Dict:
        """Convert PriceData to dictionary."""
        return {
            'symbol': self.symbol,
            'date': self.date,
            'open': self.open,
            'high': self.high,
            'low': self.low,
            'close': self.close,
            'volume': self.volume,
            'adjusted_close': self.adjusted_close,
            'source': self.source
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PriceData':
        """Create PriceData from dictionary."""
        return cls(**data)


@dataclass
class Portfolio:
    """
    Portfolio structure for holding multiple assets.
    
    Attributes:
        name: Portfolio name
        assets: Dictionary mapping symbols to their allocations (weights)
        initial_value: Initial portfolio value in dollars
        data: Historical price data for all assets
        created_at: Portfolio creation timestamp
    """
    name: str
    assets: Dict[str, float]  # symbol -> weight
    initial_value: float
    data: Optional[pd.DataFrame] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Validate portfolio after initialization."""
        if not self.assets:
            raise ValueError("Portfolio must contain at least one asset")
        
        total_weight = sum(self.assets.values())
        if not 0.99 <= total_weight <= 1.01:  # Allow small rounding errors
            raise ValueError(f"Asset weights must sum to 1.0, got {total_weight}")
        
        if self.initial_value <= 0:
            raise ValueError("Initial value must be positive")
    
    def get_asset_value(self, symbol: str) -> float:
        """Get the initial value allocated to a specific asset."""
        if symbol not in self.assets:
            raise KeyError(f"Asset {symbol} not in portfolio")
        return self.initial_value * self.assets[symbol]
    
    def to_dict(self) -> Dict:
        """Convert Portfolio to dictionary."""
        return {
            'name': self.name,
            'assets': self.assets,
            'initial_value': self.initial_value,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class SimulationResult:
    """
    Results from a Monte Carlo portfolio simulation.
    
    Attributes:
        portfolio_name: Name of the portfolio
        num_simulations: Number of simulation runs
        time_horizon: Simulation time horizon in days
        simulated_paths: Array of simulated portfolio value paths
        statistics: Dictionary of statistical metrics
        final_values: Array of final portfolio values
    """
    portfolio_name: str
    num_simulations: int
    time_horizon: int
    simulated_paths: pd.DataFrame
    statistics: Dict[str, float]
    final_values: List[float]
    
    def get_percentile(self, percentile: float) -> float:
        """Get a specific percentile of final values."""
        return pd.Series(self.final_values).quantile(percentile / 100)
    
    def get_var(self, confidence: float = 95) -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            confidence: Confidence level (e.g., 95 for 95% VaR)
        
        Returns:
            VaR value
        """
        return self.get_percentile(100 - confidence)
