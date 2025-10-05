"""
Tests for portfolio simulator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.financial_toolkit.data_models import Portfolio
from src.financial_toolkit.portfolio_simulator import PortfolioSimulator


class TestPortfolioSimulator:
    """Tests for PortfolioSimulator class."""

    @pytest.fixture
    def sample_portfolio_data(self):
        """Create sample portfolio with historical data."""
        dates = pd.date_range('2024-01-01', periods=100, freq='D')

        # Create multi-index dataframe
        data_list = []
        for date in dates:
            for symbol in ['AAPL', 'GOOGL']:
                # Create realistic price movement
                base_price = 100 if symbol == 'AAPL' else 150
                random_change = np.random.normal(0, 0.02)
                price = base_price * (1 + random_change)

                data_list.append({
                    'symbol': symbol,
                    'open': price * 0.99,
                    'high': price * 1.01,
                    'low': price * 0.98,
                    'close': price,
                    'volume': 1000000
                })

        df = pd.DataFrame(data_list)
        df['date'] = pd.concat([pd.Series(dates)] * 2).reset_index(drop=True)
        df = df.set_index(['date', 'symbol']).sort_index()

        portfolio = Portfolio(
            name='Test Portfolio',
            assets={'AAPL': 0.6, 'GOOGL': 0.4},
            initial_value=10000.0,
            data=df
        )

        return portfolio

    def test_simulator_initialization(self, sample_portfolio_data):
        """Test initializing the simulator."""
        simulator = PortfolioSimulator(sample_portfolio_data)

        assert simulator.portfolio == sample_portfolio_data
        assert simulator.returns is None

    def test_prepare_data(self, sample_portfolio_data):
        """Test preparing data for simulation."""
        simulator = PortfolioSimulator(sample_portfolio_data)
        simulator.prepare_data()

        assert simulator.returns is not None
        assert simulator.mean_returns is not None
        assert simulator.cov_matrix is not None
        assert len(simulator.mean_returns) == 2  # Two assets

    def test_simulate(self, sample_portfolio_data):
        """Test running Monte Carlo simulation."""
        simulator = PortfolioSimulator(sample_portfolio_data)

        result = simulator.simulate(
            num_simulations=100,
            time_horizon=30,
            random_seed=42
        )

        assert result.portfolio_name == 'Test Portfolio'
        assert result.num_simulations == 100
        assert result.time_horizon == 30
        assert len(result.final_values) == 100
        assert result.simulated_paths.shape == (30, 100)

    def test_simulate_statistics(self, sample_portfolio_data):
        """Test simulation statistics."""
        simulator = PortfolioSimulator(sample_portfolio_data)

        result = simulator.simulate(
            num_simulations=1000,
            time_horizon=50,
            random_seed=42
        )

        stats = result.statistics
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'min' in stats
        assert 'max' in stats
        assert 'var_95' in stats
        assert 'expected_return' in stats
        assert 'probability_of_loss' in stats

        # Check that statistics are reasonable
        assert stats['mean'] > 0
        assert stats['std'] > 0
        assert 0 <= stats['probability_of_loss'] <= 1

    def test_calculate_risk_metrics(self, sample_portfolio_data):
        """Test calculating risk metrics."""
        simulator = PortfolioSimulator(sample_portfolio_data)

        risk_metrics = simulator.calculate_risk_metrics()

        assert 'volatility' in risk_metrics
        assert 'sharpe_ratio' in risk_metrics
        assert 'max_drawdown' in risk_metrics
        assert 'var_95' in risk_metrics
        assert 'cvar_95' in risk_metrics

        # Check that metrics are reasonable
        assert risk_metrics['volatility'] > 0
        assert risk_metrics['max_drawdown'] <= 0  # Drawdown is negative

    def test_simulate_without_data(self):
        """Test simulation fails without data."""
        portfolio = Portfolio(
            name='No Data Portfolio',
            assets={'AAPL': 1.0},
            initial_value=10000.0
        )

        simulator = PortfolioSimulator(portfolio)

        with pytest.raises(ValueError, match="must have historical data"):
            simulator.simulate()

    def test_calculate_optimal_weights(self, sample_portfolio_data):
        """Test calculating optimal weights."""
        simulator = PortfolioSimulator(sample_portfolio_data)

        optimal_weights = simulator.calculate_optimal_weights(method='sharpe')

        assert isinstance(optimal_weights, dict)
        assert 'AAPL' in optimal_weights
        assert 'GOOGL' in optimal_weights
        assert abs(sum(optimal_weights.values()) - 1.0) < 1e-6

    def test_simulation_reproducibility(self, sample_portfolio_data):
        """Test that simulation is reproducible with same seed."""
        simulator1 = PortfolioSimulator(sample_portfolio_data)
        result1 = simulator1.simulate(num_simulations=50, time_horizon=20, random_seed=42)

        simulator2 = PortfolioSimulator(sample_portfolio_data)
        result2 = simulator2.simulate(num_simulations=50, time_horizon=20, random_seed=42)

        assert result1.final_values == result2.final_values
