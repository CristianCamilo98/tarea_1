"""
Data Models Module

Defines standardized dataclasses for financial market data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from statistics import mean, stdev
import matplotlib.pyplot as plt
import seaborn as sns
import sys


@dataclass
class PriceData:
    """
    Standardized price data structure for financial instruments.

    Attributes:
        symbol: Ticker symbol (e.g., 'AAPL', 'GOOGL')
        date: Date of the price data, needs to be passed in UTC timezone
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

    def __post_init__(self):
        """
        Cleaning and validation after initialization

        """
        # Verify datetime is in UTC
        if self.date.tzinfo is None or self.date.tzinfo.utcoffset(self.date) is None:
            raise ValueError("date must be timezone-aware and in UTC")

        # Basic validation
        if self.open <= 0 or self.high <= 0 or self.low <= 0 or self.close <= 0:
            raise ValueError("OHLC prices must be > 0")
        if self.volume < 0:
            raise ValueError("volume must be >= 0")
        if self.high < self.low:
            raise ValueError("high must be >= low")

        tol = 1e-8
        if not (self.low - tol <= self.open <= self.high + tol):
            raise ValueError(f"open ({self.open}) must be within [low, high] (±{tol})")
        if not (self.low - tol <= self.close <= self.high + tol):
            raise ValueError(f"close ({self.close}) must be within [low, high] (±{tol})")



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
class DividendData:
    """
    Standardized dividend data structure for financial instruments.

    Attributes:
        symbol: Ticker symbol (e.g., 'AAPL', 'GOOGL')
        date: Date of the dividend payment
        dividend: Dividend amount
        source: Data source identifier (e.g., 'yahoo', 'alpha_vantage')
    """
    symbol: str
    date: datetime
    dividend: float
    source: str = "unknown"

    def to_dict(self) -> Dict:
        """Convert DividendData to dictionary."""
        return {
            'symbol': self.symbol,
            'date': self.date,
            'dividend': self.dividend,
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'DividendData':
        """Create DividendData from dictionary."""
        return cls(**data)





@dataclass
class PriceSeriesData:
    """
    Collection of price data with automatic statistical calculations.

    Attributes:
        prices: List of PriceData objects
        mean_price: Automatically calculated mean closing price
        std_dev: Automatically calculated standard deviation of closing prices
        statistics: Dictionary containing additional statistical measures
    """
    prices: List[PriceData]
    mean_price: Optional[float] = field(init=False, default=None)
    std_dev: Optional[float] = field(init=False, default=None)
    statistics: Dict[str, float] = field(init=False, default_factory=dict)


    def __post_init__(self):
        """Automatically calculate basic statistics after initialization."""
        if not self.prices:
            raise ValueError("PriceSeriesData must contain at least one price data point")

        # Validate all prices belong to the same symbol
        symbols = {price.symbol for price in self.prices}
        if len(symbols) > 1:
            raise ValueError(f"All prices must belong to the same symbol. Found: {symbols}")

        self._validation_post_init()
        self._calculate_basic_statistics()
        self._calculate_extended_statistics()

    def _validation_post_init(self):
        """
        Cleaning and validation after initialization

        """
        # Verify no diuplicate dates
        dates = [price.date for price in self.prices]
        if len(dates) != len(set(dates)):
            raise ValueError("Duplicate dates found in PriceSeriesData of symbol {self.symbol}")

    @property
    def symbol(self) -> str:
        """Get the symbol from the price data."""
        return self.prices[0].symbol

    def _calculate_basic_statistics(self):
        """Calculate mean and standard deviation automatically."""
        closing_prices = [price.close for price in self.prices]

        if len(closing_prices) >= 1:
            self.mean_price = mean(closing_prices)
        else:
            self.mean_price = None

        if len(closing_prices) >= 2:
            self.std_dev = stdev(closing_prices)
        else:
            self.std_dev = None

    def _calculate_extended_statistics(self):
        """Calculate additional statistical measures."""
        volumes = [price.volume for price in self.prices]
        min_price = min([price.low for price in self.prices])
        max_price = max([price.high for price in self.prices])
        closing_prices = [price.close for price in self.prices]

        if len(self.prices) >= 1:
            self.statistics.update({
                'min_open': min([price.open for price in self.prices]),
                'min_close': min([price.close for price in self.prices]),
                'min_price': min_price,
                'max_open': max([price.open for price in self.prices]),
                'max_close': max([price.close for price in self.prices]),
                'max_price': max_price,
                'max_price_range': max_price - min_price,
                'total_volume': sum(volumes),
                'avg_volume': mean(volumes),
            })

        if len(closing_prices) >= 2:
            # Calculate daily returns
            returns = [
                (closing_prices[i] - closing_prices[i-1]) / closing_prices[i-1]
                for i in range(1, len(closing_prices))
            ]

            if returns:
                self.statistics.update({
                    'avg_return': mean(returns),
                    'return_volatility': stdev(returns) if len(returns) >= 2 else 0,
                    'min_return': min(returns),
                    'max_return': max(returns),
                })

    def get_volatility(self) -> float:
        """Get price volatility (annualized standard deviation)."""
        if self.std_dev is None:
            return 0.0
        # Assuming daily data, annualize with sqrt(252)
        return self.std_dev * np.sqrt(252)

    def get_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio
        """
        avg_return = self.statistics.get('avg_return', 0)
        return_volatility = self.statistics.get('return_volatility', 0)

        if return_volatility == 0:
            return 0.0

        # Convert to annualized values
        annual_return = avg_return * 252
        annual_volatility = return_volatility * np.sqrt(252)

        return (annual_return - risk_free_rate) / annual_volatility

    def get_price_percentile(self, percentile: float) -> np.float64:
        """
        Get price at specific percentile.
        Args:
            percentile: Percentile to calculate (0-100)
        Returns:
            Price at the given percentile
        """
        closing_prices = [price.close for price in self.prices]
        return np.percentile(closing_prices, percentile)

    def get_moving_average(self, window: int) -> pd.DataFrame:
        """
        Calculate moving average with specified window.
        Args:
            window: Window size for moving average
        Returns:
            DataFrame with date as index and moving average as column
        """
        date_to_price = {}
        for price in self.prices:
            date_to_price[price.date] = price.close

        series = pd.Series(date_to_price).sort_index()
        moving_avg_series = series.rolling(window=window).mean()
        return moving_avg_series.to_frame(name=f'moving_average_{window}')


    def add_prices(self, price_data: list[PriceData]):
        """Add a new price data point and recalculate statistics."""

        self.prices.extend(price_data)
        # print("length of prices after adding:", len(self.prices), file=sys.stderr)
        # print("first elment date:", self.prices[0].date, file=sys.stderr)
        # print("last elment date:", self.prices[-1].date, file=sys.stderr)
        # Recalculate statistics
        self._calculate_basic_statistics()
        self._calculate_extended_statistics()

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert price series to pandas DataFrame.
        Returns:
            DataFrame with date as index and price attributes as columns
        """
        data: List[Dict] = [
                price.to_dict()
                for price in self.prices
        ]

        df = pd.DataFrame(data)
        df = df.sort_values(by='date')
        if not df.empty:
            df.set_index('date', inplace=True)
        return df

    def get_statistics_summary(self) -> Dict[str, float]:
        """
        Get a comprehensive summary of all calculated statistics.
        Returns:
            Dictionary of key statistics
        """
        summary = {
            'mean_price': self.mean_price or 0,
            'std_dev': self.std_dev or 0,
            'volatility_annualized': self.get_volatility(),
            'sharpe_ratio': self.get_sharpe_ratio(),
        }
        summary.update(self.statistics)
        return summary

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
        tickers: List of ticker symbols in the portfolio
        corr: Correlation matrix of asset returns
    """
    portfolio_name: str
    num_simulations: int
    time_horizon: int
    simulated_paths: np.ndarray
    statistics: Dict[str, float]
    final_values: List[float]
    tickers: List[str]
    corr: pd.DataFrame

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

    def get_statistics_summary(self) -> Dict[str, float]:
        """
        Get a summary of key statistics from the simulation.

        Returns:
            Dictionary of key statistics
        """
        summary = {
            'mean_final_value': mean(self.final_values),
            'std_dev_final_value': stdev(self.final_values) if len(self.final_values) >= 2 else 0,
            'min_final_value': min(self.final_values),
            'max_final_value': max(self.final_values),
            'VaR95': self.get_var(95),
            'VaR99': self.get_var(99),
        }
        return summary

    def plots_report(self, location='reports'):
        """
        Visualize the simulation results using seaborn for enhanced aesthetics.
        Args:
          location: Path to save the plots (default 'reports')
        Returns:
          None
        """
        # Set seaborn style for better aesthetics
        sns.set_style("whitegrid")
        sns.set_palette("husl")

        # Create subplots for comprehensive visualization a 4x4 grid of plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Monte Carlo Simulation Analysis: {self.portfolio_name}', fontsize=16, fontweight='bold')

        # 1. Simulation paths plot
        ax1 = axes[0, 0]
        paths_df = pd.DataFrame(self.simulated_paths.T)

        # Plot a sample of paths to avoid overcrowding
        sample_size = min(150, self.num_simulations)
        sample_indices = np.random.choice(self.num_simulations, sample_size, replace=False)

        for i in sample_indices:
            ax1.plot(paths_df.iloc[:, i], alpha=0.3, linewidth=0.8)

        # Add median path
        median_path = paths_df.median(axis=1)
        ax1.plot(median_path, color='red', linewidth=2, label='Median Path')

        # Add percentile bands
        p5 = paths_df.quantile(0.05, axis=1)
        p95 = paths_df.quantile(0.95, axis=1)
        ax1.fill_between(range(len(p5)), p5, p95, alpha=0.2, color='yellow', label='5th-95th Percentile')

        ax1.set_title('Portfolio Value Simulation Paths', fontweight='bold')
        ax1.set_xlabel('Days')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Final values distribution
        ax2 = axes[0, 1]
        sns.histplot(self.final_values, bins=75, kde=True, ax=ax2, alpha=0.7)
        ax2.axvline(np.median(self.final_values), color='red', linestyle='--',
                   label=f'Median: ${np.median(self.final_values):,.0f}')
        ax2.axvline(np.percentile(self.final_values, 5), color='orange', linestyle='--',
                   label=f'5th Percentile: ${np.percentile(self.final_values, 5):,.0f}')
        ax2.axvline(np.percentile(self.final_values, 95), color='green', linestyle='--',
                   label=f'95th Percentile: ${np.percentile(self.final_values, 95):,.0f}')
        ax2.set_title('Distribution of Final Portfolio Values', fontweight='bold')
        ax2.set_xlabel('Final Portfolio Value ($)')
        ax2.set_ylabel('Frequency')
        ax2.legend()

        # 3. Box plot of final values
        ax3 = axes[1, 0]
        sns.boxplot(y=self.final_values, ax=ax3)
        ax3.set_title('Final Portfolio Values - Box Plot', fontweight='bold')
        ax3.set_ylabel('Portfolio Value ($)')

        # 4. Risk metrics visualization
        ax4 = axes[1, 1]
        risk_metrics = {
            'VaR 95%': self.get_var(95),
            'VaR 99%': self.get_var(99),
            'Mean': np.mean(self.final_values),
            'Median': np.median(self.final_values)
        }

        bars = ax4.bar(risk_metrics.keys(), risk_metrics.values(),
                      color=sns.color_palette("viridis", len(risk_metrics)))
        ax4.set_title('Key Risk Metrics', fontweight='bold')
        ax4.set_ylabel('Portfolio Value ($)')

        # Add value labels on bars
        for bar, value in zip(bars, risk_metrics.values()):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${value:,.0f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_to_save = f"{location}/monte_carlo_simulation_{timestamp}.pdf"
        plt.savefig(path_to_save, dpi=300)
        print(f"Simulation plots saved to {path_to_save}")
        plt.show()

    def plots_detailed_report(self, location='reports'):
        """
        Create additional detailed visualizations using seaborn.
        Args:
          location: Path to save the plots (default 'reports')
        Returns:
          None
        """

        sns.set_style("whitegrid")

        # Create a comprehensive analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        fig.suptitle(f'Detailed Monte Carlo Analysis: {self.portfolio_name}', fontsize=18, fontweight='bold')

        # 1. Violin plot of final values
        ax1 = axes[0, 0]
        sns.violinplot(y=self.final_values, ax=ax1, inner='quartile', color='skyblue')
        ax1.set_title('Final Values Distribution (Violin Plot)', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')

        # 2. Cumulative distribution of final values
        ax2 = axes[0, 1]
        sorted_values = np.sort(self.final_values)
        cumulative_prob = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
        sns.lineplot(x=sorted_values, y=cumulative_prob, ax=ax2)
        median = np.median(self.final_values)
        ax2.axvline(median, color='green', linestyle=':', label=f'Median: ${median:,.0f}')
        ax2.set_title('Cumulative Distribution Function', fontweight='bold')
        ax2.set_xlabel('Portfolio Value ($)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 3. Returns distribution
        ax3 = axes[1, 0]
        daily_returns = []
        for path in self.simulated_paths:
            returns = np.diff(path) / path[:-1]
            daily_returns.extend(returns)

        sns.histplot(daily_returns, bins=50, kde=True, ax=ax3, alpha=0.7)
        ax3.set_title('Simulated Daily Returns Distribution', fontweight='bold')
        ax3.set_xlabel('Daily Return')
        ax3.set_ylabel('Frequency')


        # 4. Heat map of correlation in returns (if multiple assets)
        ax4 = axes[1, 1]
        if self.corr.shape[0] > 1:
            sns.heatmap(self.corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax4)
            ax4.set_title('Correlation Matrix of Asset Returns', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Not enough assets for correlation matrix',
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=12, color='red')
            ax4.set_title('Correlation Matrix of Asset Returns', fontweight='bold')
            ax4.axis('off')

        plt.tight_layout()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path_to_save = f"{location}/detailed_monte_carlo_detailed_analysis_{timestamp}.pdf"
        print(f"Detailed analysis plots saved to {path_to_save}")
        plt.savefig(path_to_save, dpi=300)
        plt.show()



@dataclass
class Portfolio:
    """
    A theortical/analytical portfolio to perform simulations on.
    Attributes:
        holdings: Dictionary mapping ticker symbols to PriceSeriesData
        weights: Dictionary mapping ticker symbols to portfolio weights
        base_currency: Base currency for the portfolio (default 'USD')
        risk_free_rate: Annual risk-free rate for calculations (default 2%)
        price_field: Price field to use for calculations (default 'close')
        tz: Timezone for date handling (default 'UTC')
    """
    holdings: Dict[str, PriceSeriesData]
    weights: Dict[str, float]
    base_currency: str = "USD"
    risk_free_rate: float = 0.02
    price_field: str = "Close"
    tz: str = "UTC"

    def __post_init__(self):
        """Validate weights and holdings after initialization."""
        if not self.holdings:
            raise ValueError("Portfolio must contain at least one holding")

        if set(self.weights.keys()) != set(self.holdings.keys()):
            raise ValueError("Weights keys must match holdings symbols")

        total_weight = sum(self.weights.values())
        if not np.isclose(total_weight, 1.0):
            raise ValueError(f"Weights must sum to 1.0, got {total_weight}")

        for symbol, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Weight for {symbol} must be non-negative, got {weight}")

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Portfolio.

        This representation shows the portfolio holdings, weights, and base currency.
        """
        holdings_str = '\n'.join([
            f"  {symbol}: {weight:.2%}"
            for symbol, weight in self.weights.items()
        ])

        return f"Portfolio ({self.base_currency}):\n{holdings_str}"

    def _compute_correlation_maxtrix(self, sigma: np.ndarray, tickers: List[str]) -> pd.DataFrame:
        """
        Compute correlation matrix from covariance matrix.
        Args:
            sigma: Covariance matrix
            tickers: List of ticker symbols
        Returns:
            Correlation matrix as DataFrame
        """
        std = np.sqrt(np.diag(sigma))
        # guard against division by zero
        std[std == 0] = np.nan
        corr_matrix = sigma / np.outer(std, std)
        np.fill_diagonal(corr_matrix, 1.0)
        # Numerical cleanup
        corr_matrix = np.clip(corr_matrix, -1.0, 1.0)
        corr_df = pd.DataFrame(corr_matrix, index=tickers, columns=tickers)
        return corr_df

    def monte_carlo_simulation(
        self,
        num_simulations: int = 1000,
        horizon_days: int = 252,
        seed: Optional[int] = None,
        method: str = 'gaussian',
        V0: float = 10000.0
    ) -> SimulationResult:
        """
        Perform Monte Carlo simulation on the portfolio.
        Args:
            num_simulations: Number of simulation runs (default 1000)
            horizon_days: Simulation time horizon in days (default 252)
            seed: Random seed for reproducibility (default None)
            method: Simulation method ('gaussian' or 'bootstrap', default 'gaussian')
            V0: Initial portfolio value (default 10000.0)
        Returns:
            SimulationResult containing simulation details and statistics
        """
        if seed is not None:
            np.random.default_rng(seed)
        else:
            np.random.default_rng()

        rng = np.random.default_rng(seed)

        # 1) Extract CLOSE columns
        price_data = {symbol: { 'priceseriesdata': priceseriesdata.to_dataframe(), 'weight': self.weights[symbol] } for symbol, priceseriesdata in self.holdings.items()}
        tickers = list(price_data.keys())
        closes = pd.concat([price_data[ticket]["priceseriesdata"]["adjusted_close"].rename(ticket).astype(float) for ticket in tickers],axis=1, join="inner")


        log_rets = np.log(closes / closes.shift(1)).dropna()   # [T, N]
        T, _ = log_rets.shape

        # 2) Weights in the same order
        weights = np.array([price_data[t]["weight"] for t in tickers], dtype=float)

        # 3) Daily μ and Σ (joint, across assets)
        X = log_rets.to_numpy()
        mu = X.mean(axis=0)                             # (N,)
        # The covariance matrix measures how different assets move together (the relationship between assets)
        sigma = np.cov(X, rowvar=False, ddof=1)         # (N, N)
        correlation_matrix = self._compute_correlation_maxtrix(sigma, tickers)


        # 4) Simulate rebalanced portfolio
        H = int(horizon_days)
        if method.lower() == "gaussian":
            per_day = rng.multivariate_normal(mu, sigma, size=(num_simulations, H))   # [n_sims, H, N]
        elif method.lower() == "bootstrap":
            idx = rng.integers(0, T, size=(num_simulations, H))
            per_day = X[idx]                                                 # [n_sims, H, N]
        else:
            raise ValueError("method must be 'gaussian' or 'bootstrap'")

        # r_p_days is the portfolio return for each day in each simulation (n_sims, days)
        r_p_days = per_day @ weights
        # Simulated_paths is the portfolio value path for each simulation (n_sims, days)
        simulated_paths = V0 * np.exp(np.cumsum(r_p_days, axis=1))
        # R_H is the total return over the horizon for each simulation (n_sims,)
        R_H = r_p_days.sum(axis=1)
        # V_H is the portfolio value at the end of the horizon for each simulation
        V_H = V0 * np.exp(R_H)

        losses = V0 - V_H
        # VaR95 is the 95th percentile of losses
        VaR95 = float(np.quantile(losses, 0.95))
        tail = losses[losses >= VaR95]
        # ES95 is the expected shortfall at 95% confidence level
        ES95 = float(tail.mean() if tail.size else VaR95)

        result = SimulationResult(
               portfolio_name="Portfolio",
               num_simulations=num_simulations,
               time_horizon=horizon_days,
               simulated_paths=simulated_paths,
               statistics={
                 "E[R]": float(R_H.mean()),
                 "Std[R]": float(R_H.std(ddof=1)),
                 "VaR95": VaR95,
                 "ES95": ES95,
               },
               final_values=V_H.tolist(),
               tickers=tickers,
               corr=correlation_matrix
         )

        return result

@dataclass
class SplitsData:
    """
    Standardized stock splits data structure for financial instruments.

    Attributes:
        symbol: Ticker symbol (e.g., 'AAPL', 'GOOGL')
        data: Series of stock splits with date as index and split ratio as values
        source: Data source identifier (e.g., 'yahoo', 'alpha_vantage')
    """

    symbol: str
    data: pd.Series
    source: str = "unknown"

    def to_dict(self) -> Dict:
        """Convert SplitsData to dictionary."""
        return {
            'symbol': self.symbol,
            'data': self.data.to_dict(orient='records'),
            'source': self.source
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'SplitsData':
        """Create SplitsData from dictionary."""
        df = pd.DataFrame(data['data'])
        return cls(symbol=data['symbol'], data=df, source=data.get('source', 'unknown'))

