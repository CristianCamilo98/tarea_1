"""
Tests for data cleaner.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.financial_toolkit.data_cleaner import DataCleaner


class TestDataCleaner:
    """Tests for DataCleaner class."""

    @pytest.fixture
    def sample_data(self):
        """Create sample financial data for testing."""
        dates = pd.date_range('2024-01-01', periods=10, freq='D')
        data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
            'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
            'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
        }, index=dates)
        return data

    def test_remove_duplicates(self, sample_data):
        """Test removing duplicate rows."""
        # Add duplicate rows
        data_with_dupes = pd.concat([sample_data, sample_data.iloc[[0, 1]]])

        cleaner = DataCleaner()
        cleaned = cleaner.remove_duplicates(data_with_dupes)

        assert len(cleaned) == len(sample_data)
        assert not cleaned.index.duplicated().any()

    def test_handle_missing_values_ffill(self, sample_data):
        """Test forward fill for missing values."""
        data_with_missing = sample_data.copy()
        data_with_missing.loc[data_with_missing.index[5], 'close'] = np.nan

        cleaner = DataCleaner()
        cleaned = cleaner.handle_missing_values(data_with_missing, method='ffill')

        assert not cleaned['close'].isna().any()
        assert cleaned.loc[data_with_missing.index[5], 'close'] == sample_data.loc[sample_data.index[4], 'close']

    def test_handle_missing_values_drop(self, sample_data):
        """Test dropping missing values."""
        data_with_missing = sample_data.copy()
        data_with_missing.loc[data_with_missing.index[5], 'close'] = np.nan

        cleaner = DataCleaner()
        cleaned = cleaner.handle_missing_values(data_with_missing, method='drop')

        assert len(cleaned) == len(sample_data) - 1
        assert not cleaned.isna().any().any()

    def test_validate_price_data(self, sample_data):
        """Test price data validation."""
        cleaner = DataCleaner()
        cleaned = cleaner.validate_price_data(sample_data)

        # All rows should pass validation
        assert len(cleaned) == len(sample_data)

    def test_validate_price_data_invalid(self):
        """Test price data validation with invalid data."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        invalid_data = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 99, 104, 105, 106],  # Invalid: high < low for index 1
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
        }, index=dates)

        cleaner = DataCleaner()
        cleaned = cleaner.validate_price_data(invalid_data)

        # Invalid row should be removed
        assert len(cleaned) < len(invalid_data)

    def test_calculate_returns_simple(self, sample_data):
        """Test calculating simple returns."""
        cleaner = DataCleaner()
        data_with_returns = cleaner.calculate_returns(sample_data, method='simple')

        assert 'returns' in data_with_returns.columns
        assert data_with_returns['returns'].iloc[0] is pd.NA or np.isnan(data_with_returns['returns'].iloc[0])

        # Check calculation
        expected_return = (sample_data['close'].iloc[1] - sample_data['close'].iloc[0]) / sample_data['close'].iloc[0]
        assert abs(data_with_returns['returns'].iloc[1] - expected_return) < 1e-10

    def test_calculate_returns_log(self, sample_data):
        """Test calculating log returns."""
        cleaner = DataCleaner()
        data_with_returns = cleaner.calculate_returns(sample_data, method='log')

        assert 'returns' in data_with_returns.columns
        assert data_with_returns['returns'].iloc[0] is pd.NA or np.isnan(data_with_returns['returns'].iloc[0])

        # Check calculation
        expected_return = np.log(sample_data['close'].iloc[1] / sample_data['close'].iloc[0])
        assert abs(data_with_returns['returns'].iloc[1] - expected_return) < 1e-10

    def test_clean_data_full_pipeline(self, sample_data):
        """Test full cleaning pipeline."""
        # Add some issues
        data_with_issues = sample_data.copy()
        data_with_issues.loc[data_with_issues.index[5], 'close'] = np.nan

        cleaner = DataCleaner()
        cleaned = cleaner.clean_data(
            data_with_issues,
            remove_duplicates=True,
            handle_missing='ffill',
            validate_prices=True,
            calculate_returns=True
        )

        assert 'returns' in cleaned.columns
        assert not cleaned['close'].isna().any()
        assert len(cleaned) > 0

    def test_remove_outliers_iqr(self, sample_data):
        """Test removing outliers using IQR method."""
        # Add outlier
        data_with_outlier = sample_data.copy()
        data_with_outlier.loc[data_with_outlier.index[0], 'close'] = 1000  # Outlier

        cleaner = DataCleaner()
        cleaned = cleaner.remove_outliers(data_with_outlier, 'close', method='iqr', threshold=1.5)

        # Outlier should be removed
        assert len(cleaned) < len(data_with_outlier)
        assert 1000 not in cleaned['close'].values
