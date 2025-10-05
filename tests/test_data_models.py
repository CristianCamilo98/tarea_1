"""
Tests for data models.
"""

import pytest
from datetime import datetime
import pandas as pd
from src.financial_toolkit.data_models import PriceData, Portfolio, SimulationResult


class TestPriceData:
    """Tests for PriceData dataclass."""

    def test_price_data_creation(self):
        """Test creating a PriceData object."""
        price_data = PriceData(
            symbol='AAPL',
            date=datetime(2024, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000,
            adjusted_close=154.0,
            source='yahoo'
        )

        assert price_data.symbol == 'AAPL'
        assert price_data.close == 154.0
        assert price_data.source == 'yahoo'

    def test_price_data_to_dict(self):
        """Test converting PriceData to dictionary."""
        price_data = PriceData(
            symbol='AAPL',
            date=datetime(2024, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )

        data_dict = price_data.to_dict()
        assert isinstance(data_dict, dict)
        assert data_dict['symbol'] == 'AAPL'
        assert data_dict['close'] == 154.0

    def test_price_data_from_dict(self):
        """Test creating PriceData from dictionary."""
        data_dict = {
            'symbol': 'GOOGL',
            'date': datetime(2024, 1, 1),
            'open': 100.0,
            'high': 105.0,
            'low': 99.0,
            'close': 104.0,
            'volume': 500000
        }

        price_data = PriceData.from_dict(data_dict)
        assert price_data.symbol == 'GOOGL'
        assert price_data.close == 104.0


class TestPortfolio:
    """Tests for Portfolio dataclass."""

    def test_portfolio_creation(self):
        """Test creating a Portfolio object."""
        portfolio = Portfolio(
            name='Test Portfolio',
            assets={'AAPL': 0.5, 'GOOGL': 0.5},
            initial_value=10000.0
        )

        assert portfolio.name == 'Test Portfolio'
        assert len(portfolio.assets) == 2
        assert portfolio.initial_value == 10000.0

    def test_portfolio_weight_validation(self):
        """Test portfolio weight validation."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            Portfolio(
                name='Invalid Portfolio',
                assets={'AAPL': 0.5, 'GOOGL': 0.3},
                initial_value=10000.0
            )

    def test_portfolio_empty_assets(self):
        """Test portfolio with no assets."""
        with pytest.raises(ValueError, match="at least one asset"):
            Portfolio(
                name='Empty Portfolio',
                assets={},
                initial_value=10000.0
            )

    def test_portfolio_negative_value(self):
        """Test portfolio with negative initial value."""
        with pytest.raises(ValueError, match="must be positive"):
            Portfolio(
                name='Negative Portfolio',
                assets={'AAPL': 1.0},
                initial_value=-1000.0
            )

    def test_get_asset_value(self):
        """Test getting asset value."""
        portfolio = Portfolio(
            name='Test Portfolio',
            assets={'AAPL': 0.6, 'GOOGL': 0.4},
            initial_value=10000.0
        )

        assert portfolio.get_asset_value('AAPL') == 6000.0
        assert portfolio.get_asset_value('GOOGL') == 4000.0

    def test_get_asset_value_invalid_symbol(self):
        """Test getting asset value for invalid symbol."""
        portfolio = Portfolio(
            name='Test Portfolio',
            assets={'AAPL': 1.0},
            initial_value=10000.0
        )

        with pytest.raises(KeyError):
            portfolio.get_asset_value('INVALID')

    def test_portfolio_to_dict(self):
        """Test converting Portfolio to dictionary."""
        portfolio = Portfolio(
            name='Test Portfolio',
            assets={'AAPL': 0.5, 'GOOGL': 0.5},
            initial_value=10000.0
        )

        portfolio_dict = portfolio.to_dict()
        assert isinstance(portfolio_dict, dict)
        assert portfolio_dict['name'] == 'Test Portfolio'
        assert portfolio_dict['initial_value'] == 10000.0


class TestSimulationResult:
    """Tests for SimulationResult dataclass."""

    def test_simulation_result_creation(self):
        """Test creating a SimulationResult object."""
        sim_data = pd.DataFrame({
            'Simulation_1': [10000, 10100, 10200],
            'Simulation_2': [10000, 9900, 9800]
        })

        result = SimulationResult(
            portfolio_name='Test Portfolio',
            num_simulations=2,
            time_horizon=3,
            simulated_paths=sim_data,
            statistics={'mean': 10000},
            final_values=[10200, 9800]
        )

        assert result.portfolio_name == 'Test Portfolio'
        assert result.num_simulations == 2
        assert len(result.final_values) == 2

    def test_get_percentile(self):
        """Test getting percentile of final values."""
        result = SimulationResult(
            portfolio_name='Test',
            num_simulations=5,
            time_horizon=1,
            simulated_paths=pd.DataFrame(),
            statistics={},
            final_values=[9000, 9500, 10000, 10500, 11000]
        )

        p50 = result.get_percentile(50)
        assert p50 == 10000

    def test_get_var(self):
        """Test calculating VaR."""
        result = SimulationResult(
            portfolio_name='Test',
            num_simulations=100,
            time_horizon=1,
            simulated_paths=pd.DataFrame(),
            statistics={},
            final_values=list(range(9000, 11000, 20))
        )

        var_95 = result.get_var(95)
        assert isinstance(var_95, float)
        assert var_95 < 10000  # VaR should be below mean
