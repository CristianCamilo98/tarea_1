"""
Report Generator Module

Generates markdown reports for financial analysis.
"""

from typing import Optional, Dict
from datetime import datetime
import pandas as pd
from .data_models import Portfolio, SimulationResult


class ReportGenerator:
    """
    Generates markdown reports for financial analysis.
    """
    
    def __init__(self):
        """Initialize the report generator."""
        pass
    
    @staticmethod
    def generate_portfolio_report(portfolio: Portfolio) -> str:
        """
        Generate a markdown report for a portfolio.
        
        Args:
            portfolio: Portfolio object
        
        Returns:
            Markdown formatted report string
        """
        report = f"# Portfolio Report: {portfolio.name}\n\n"
        report += f"**Created:** {portfolio.created_at.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += f"**Initial Value:** ${portfolio.initial_value:,.2f}\n\n"
        
        report += "## Asset Allocation\n\n"
        report += "| Symbol | Weight | Value |\n"
        report += "|--------|--------|-------|\n"
        
        for symbol, weight in portfolio.assets.items():
            value = portfolio.get_asset_value(symbol)
            report += f"| {symbol} | {weight:.2%} | ${value:,.2f} |\n"
        
        report += "\n"
        
        return report
    
    @staticmethod
    def generate_simulation_report(result: SimulationResult) -> str:
        """
        Generate a markdown report for simulation results.
        
        Args:
            result: SimulationResult object
        
        Returns:
            Markdown formatted report string
        """
        report = f"# Monte Carlo Simulation Report: {result.portfolio_name}\n\n"
        report += f"**Number of Simulations:** {result.num_simulations:,}\n\n"
        report += f"**Time Horizon:** {result.time_horizon} days\n\n"
        
        report += "## Statistical Summary\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        stats = result.statistics
        report += f"| Mean Final Value | ${stats['mean']:,.2f} |\n"
        report += f"| Median Final Value | ${stats['median']:,.2f} |\n"
        report += f"| Standard Deviation | ${stats['std']:,.2f} |\n"
        report += f"| Minimum Value | ${stats['min']:,.2f} |\n"
        report += f"| Maximum Value | ${stats['max']:,.2f} |\n"
        report += f"| Expected Return | {stats['expected_return']:.2%} |\n"
        report += f"| Probability of Loss | {stats['probability_of_loss']:.2%} |\n"
        
        report += "\n## Risk Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        report += f"| VaR (95%) | ${stats['var_95']:,.2f} |\n"
        report += f"| VaR (99%) | ${stats['var_99']:,.2f} |\n"
        
        report += "\n"
        
        return report
    
    @staticmethod
    def generate_data_summary(data: pd.DataFrame, title: str = "Data Summary") -> str:
        """
        Generate a markdown summary of data.
        
        Args:
            data: DataFrame to summarize
            title: Report title
        
        Returns:
            Markdown formatted report string
        """
        report = f"# {title}\n\n"
        report += f"**Shape:** {data.shape[0]} rows × {data.shape[1]} columns\n\n"
        
        if hasattr(data.index, 'names') and data.index.names[0] is not None:
            report += f"**Index:** {', '.join([str(n) for n in data.index.names if n])}\n\n"
        
        report += "## Columns\n\n"
        report += "| Column | Type | Non-Null Count |\n"
        report += "|--------|------|----------------|\n"
        
        for col in data.columns:
            dtype = data[col].dtype
            non_null = data[col].notna().sum()
            report += f"| {col} | {dtype} | {non_null} |\n"
        
        report += "\n## Numeric Summary\n\n"
        
        numeric_cols = data.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            desc = data[numeric_cols].describe()
            report += desc.to_markdown() + "\n\n"
        
        return report
    
    @staticmethod
    def generate_risk_report(risk_metrics: Dict[str, float], portfolio_name: str) -> str:
        """
        Generate a markdown report for risk metrics.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            portfolio_name: Name of the portfolio
        
        Returns:
            Markdown formatted report string
        """
        report = f"# Risk Analysis Report: {portfolio_name}\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## Risk Metrics\n\n"
        report += "| Metric | Value |\n"
        report += "|--------|-------|\n"
        
        for metric, value in risk_metrics.items():
            if 'ratio' in metric.lower():
                report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
            elif 'drawdown' in metric.lower() or 'var' in metric.lower() or 'cvar' in metric.lower():
                report += f"| {metric.replace('_', ' ').title()} | {value:.4%} |\n"
            elif 'volatility' in metric.lower():
                report += f"| {metric.replace('_', ' ').title()} | {value:.4%} |\n"
            else:
                report += f"| {metric.replace('_', ' ').title()} | {value:.4f} |\n"
        
        report += "\n"
        
        return report
    
    @staticmethod
    def save_report(report: str, filename: str) -> None:
        """
        Save a report to a file.
        
        Args:
            report: Report string
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(report)
    
    def generate_full_report(
        self,
        portfolio: Portfolio,
        simulation_result: Optional[SimulationResult] = None,
        risk_metrics: Optional[Dict[str, float]] = None,
        data_summary: Optional[pd.DataFrame] = None
    ) -> str:
        """
        Generate a comprehensive report.
        
        Args:
            portfolio: Portfolio object
            simulation_result: SimulationResult object (optional)
            risk_metrics: Risk metrics dictionary (optional)
            data_summary: Data to summarize (optional)
        
        Returns:
            Markdown formatted report string
        """
        report = f"# Comprehensive Financial Analysis Report\n\n"
        report += f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        report += "---\n\n"
        
        report += self.generate_portfolio_report(portfolio)
        report += "---\n\n"
        
        if simulation_result:
            report += self.generate_simulation_report(simulation_result)
            report += "---\n\n"
        
        if risk_metrics:
            report += self.generate_risk_report(risk_metrics, portfolio.name)
            report += "---\n\n"
        
        if data_summary is not None:
            report += self.generate_data_summary(data_summary, "Historical Data Summary")
        
        return report
