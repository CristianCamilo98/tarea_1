"""
Portfolio Simulator Module

Provides Monte Carlo simulation for portfolio analysis.
"""

from typing import Optional
import numpy as np
import pandas as pd
from scipy.stats import norm
from .data_models import Portfolio, SimulationResult


class PortfolioSimulator:
    """
    Monte Carlo portfolio simulator for risk analysis.
    """

    def __init__(self, portfolio: Portfolio):
        """
        Initialize the portfolio simulator.

        Args:
            portfolio: Portfolio object to simulate
        """
        self.portfolio = portfolio
        self.returns = None
        self.mean_returns = None
        self.cov_matrix = None

    def prepare_data(self) -> None:
        """Prepare portfolio data for simulation."""
        if self.portfolio.data is None:
            raise ValueError("Portfolio must have historical data")

        # Calculate returns for each asset
        df = self.portfolio.data.copy()

        # Pivot to get each symbol as a column
        if 'symbol' in df.index.names:
            df = df.reset_index(level='symbol')
            close_prices = df.pivot_table(
                values='close',
                index=df.index,
                columns='symbol'
            )
        else:
            close_prices = df[['close']].copy()

        # Calculate returns
        close_prices = close_prices.ffill()  # Forward fill any NaN values first
        self.returns = close_prices.pct_change().dropna()

        # Calculate statistics
        self.mean_returns = self.returns.mean()
        self.cov_matrix = self.returns.cov()

    def simulate(
        self,
        num_simulations: int = 1000,
        time_horizon: int = 252,  # Trading days in a year
        random_seed: Optional[int] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for the portfolio.

        Args:
            num_simulations: Number of simulation runs
            time_horizon: Simulation time horizon in days
            random_seed: Random seed for reproducibility

        Returns:
            SimulationResult object
        """
        if self.returns is None:
            self.prepare_data()

        if random_seed is not None:
            np.random.seed(random_seed)

        # Get portfolio weights
        symbols = list(self.portfolio.assets.keys())
        weights = np.array([self.portfolio.assets[s] for s in symbols])

        # Ensure we have data for all symbols
        available_symbols = [s for s in symbols if s in self.mean_returns.index]
        if len(available_symbols) != len(symbols):
            missing = set(symbols) - set(available_symbols)
            raise ValueError(f"Missing data for symbols: {missing}")

        # Calculate portfolio statistics
        portfolio_mean = np.dot(weights, self.mean_returns[symbols])
        portfolio_std = np.sqrt(
            np.dot(weights.T, np.dot(self.cov_matrix.loc[symbols, symbols], weights))
        )

        # Run simulations
        simulations = np.zeros((time_horizon, num_simulations))
        simulations[0] = self.portfolio.initial_value

        for t in range(1, time_horizon):
            # Generate random returns
            random_returns = np.random.normal(
                portfolio_mean,
                portfolio_std,
                num_simulations
            )
            simulations[t] = simulations[t-1] * (1 + random_returns)

        # Create results DataFrame
        sim_df = pd.DataFrame(
            simulations,
            columns=[f"Simulation_{i+1}" for i in range(num_simulations)]
        )
        sim_df.index.name = 'Day'

        # Calculate statistics
        final_values = simulations[-1]
        statistics = {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std': np.std(final_values),
            'min': np.min(final_values),
            'max': np.max(final_values),
            'var_95': np.percentile(final_values, 5),
            'var_99': np.percentile(final_values, 1),
            'expected_return': (np.mean(final_values) - self.portfolio.initial_value) / self.portfolio.initial_value,
            'probability_of_loss': np.sum(final_values < self.portfolio.initial_value) / num_simulations
        }

        return SimulationResult(
            portfolio_name=self.portfolio.name,
            num_simulations=num_simulations,
            time_horizon=time_horizon,
            simulated_paths=sim_df,
            statistics=statistics,
            final_values=final_values.tolist()
        )

    def calculate_optimal_weights(
        self,
        target_return: Optional[float] = None,
        method: str = 'sharpe'
    ) -> dict:
        """
        Calculate optimal portfolio weights.

        Args:
            target_return: Target return (not used for Sharpe optimization)
            method: Optimization method ('sharpe' or 'min_variance')

        Returns:
            Dictionary of optimal weights
        """
        if self.returns is None:
            self.prepare_data()

        symbols = list(self.portfolio.assets.keys())
        n_assets = len(symbols)

        if method == 'sharpe':
            # Simple equal-weight for demonstration
            # In production, would use scipy.optimize
            weights = np.ones(n_assets) / n_assets
        elif method == 'min_variance':
            # Simple equal-weight for demonstration
            weights = np.ones(n_assets) / n_assets
        else:
            raise ValueError(f"Unknown method: {method}")

        return dict(zip(symbols, weights))

    def calculate_risk_metrics(self) -> dict:
        """
        Calculate various risk metrics for the portfolio.

        Returns:
            Dictionary of risk metrics
        """
        if self.returns is None:
            self.prepare_data()

        symbols = list(self.portfolio.assets.keys())
        weights = np.array([self.portfolio.assets[s] for s in symbols])

        # Portfolio returns
        portfolio_returns = self.returns[symbols].dot(weights)

        # Calculate metrics
        metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'sharpe_ratio': (portfolio_returns.mean() / portfolio_returns.std()) * np.sqrt(252),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'var_95': portfolio_returns.quantile(0.05),
            'cvar_95': portfolio_returns[portfolio_returns <= portfolio_returns.quantile(0.05)].mean()
        }

        return metrics

    @staticmethod
    def _calculate_max_drawdown(returns: pd.Series) -> float:
        """Calculate maximum drawdown from returns."""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
