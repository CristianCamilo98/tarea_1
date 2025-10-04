"""
Tests for report generator.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import tempfile
import os
from src.financial_toolkit.data_models import Portfolio, SimulationResult
from src.financial_toolkit.report_generator import ReportGenerator


class TestReportGenerator:
    """Tests for ReportGenerator class."""
    
    @pytest.fixture
    def sample_portfolio(self):
        """Create sample portfolio."""
        return Portfolio(
            name='Test Portfolio',
            assets={'AAPL': 0.6, 'GOOGL': 0.4},
            initial_value=10000.0
        )
    
    @pytest.fixture
    def sample_simulation_result(self):
        """Create sample simulation result."""
        sim_data = pd.DataFrame({
            f'Simulation_{i}': np.random.randn(30) * 100 + 10000
            for i in range(1, 11)
        })
        
        final_values = sim_data.iloc[-1].tolist()
        
        return SimulationResult(
            portfolio_name='Test Portfolio',
            num_simulations=10,
            time_horizon=30,
            simulated_paths=sim_data,
            statistics={
                'mean': np.mean(final_values),
                'median': np.median(final_values),
                'std': np.std(final_values),
                'min': np.min(final_values),
                'max': np.max(final_values),
                'var_95': np.percentile(final_values, 5),
                'var_99': np.percentile(final_values, 1),
                'expected_return': 0.05,
                'probability_of_loss': 0.2
            },
            final_values=final_values
        )
    
    def test_generate_portfolio_report(self, sample_portfolio):
        """Test generating portfolio report."""
        generator = ReportGenerator()
        report = generator.generate_portfolio_report(sample_portfolio)
        
        assert isinstance(report, str)
        assert 'Test Portfolio' in report
        assert '$10,000.00' in report
        assert 'AAPL' in report
        assert 'GOOGL' in report
        assert '60.00%' in report or '60.0%' in report
    
    def test_generate_simulation_report(self, sample_simulation_result):
        """Test generating simulation report."""
        generator = ReportGenerator()
        report = generator.generate_simulation_report(sample_simulation_result)
        
        assert isinstance(report, str)
        assert 'Monte Carlo Simulation' in report
        assert 'Test Portfolio' in report
        assert '10' in report  # num_simulations
        assert '30' in report  # time_horizon
        assert 'Mean Final Value' in report
    
    def test_generate_data_summary(self):
        """Test generating data summary."""
        data = pd.DataFrame({
            'close': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400]
        })
        
        generator = ReportGenerator()
        report = generator.generate_data_summary(data, title='Test Data')
        
        assert isinstance(report, str)
        assert 'Test Data' in report
        assert '5 rows' in report
        assert 'close' in report
        assert 'volume' in report
    
    def test_generate_risk_report(self):
        """Test generating risk report."""
        risk_metrics = {
            'volatility': 0.15,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.08,
            'var_95': -0.05
        }
        
        generator = ReportGenerator()
        report = generator.generate_risk_report(risk_metrics, 'Test Portfolio')
        
        assert isinstance(report, str)
        assert 'Risk Analysis' in report
        assert 'Test Portfolio' in report
        assert 'Volatility' in report
        assert 'Sharpe Ratio' in report
    
    def test_save_report(self):
        """Test saving report to file."""
        report = "# Test Report\n\nThis is a test."
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filename = os.path.join(tmpdir, 'test_report.md')
            
            generator = ReportGenerator()
            generator.save_report(report, filename)
            
            assert os.path.exists(filename)
            
            with open(filename, 'r') as f:
                content = f.read()
            
            assert content == report
    
    def test_generate_full_report(self, sample_portfolio, sample_simulation_result):
        """Test generating comprehensive report."""
        risk_metrics = {
            'volatility': 0.15,
            'sharpe_ratio': 1.5
        }
        
        data = pd.DataFrame({
            'close': [100, 101, 102],
            'volume': [1000, 1100, 1200]
        })
        
        generator = ReportGenerator()
        report = generator.generate_full_report(
            portfolio=sample_portfolio,
            simulation_result=sample_simulation_result,
            risk_metrics=risk_metrics,
            data_summary=data
        )
        
        assert isinstance(report, str)
        assert 'Comprehensive Financial Analysis' in report
        assert 'Test Portfolio' in report
        assert 'Monte Carlo' in report
        assert 'Risk Analysis' in report
        assert 'Historical Data Summary' in report
    
    def test_generate_full_report_minimal(self, sample_portfolio):
        """Test generating report with only portfolio."""
        generator = ReportGenerator()
        report = generator.generate_full_report(portfolio=sample_portfolio)
        
        assert isinstance(report, str)
        assert 'Test Portfolio' in report
        assert 'Comprehensive Financial Analysis' in report
