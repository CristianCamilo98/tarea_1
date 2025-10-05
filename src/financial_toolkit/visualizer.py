"""
Visualizer Module

Provides visualization capabilities for financial data analysis.
"""

from typing import Optional, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from .data_models import SimulationResult, Portfolio


class Visualizer:
    """
    Visualization utilities for financial data.
    """

    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize the visualizer.

        Args:
            style: Matplotlib style to use
        """
        # Use a valid style or fallback to default
        try:
            plt.style.use(style)
        except:
            # If style not available, use default
            plt.style.use('default')
        self.style = style

    @staticmethod
    def plot_price_history(
        data: pd.DataFrame,
        symbol: Optional[str] = None,
        column: str = 'close',
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot price history for a symbol.

        Args:
            data: DataFrame with price data
            symbol: Symbol to plot (if data contains multiple symbols)
            column: Column to plot
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        if symbol and 'symbol' in data.index.names:
            plot_data = data.xs(symbol, level='symbol')[column]
        else:
            plot_data = data[column]

        ax.plot(plot_data.index, plot_data.values, linewidth=2)

        if title is None:
            title = f"{symbol if symbol else 'Price'} History - {column.title()}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel(f'{column.title()} Price ($)', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_returns_distribution(
        returns: pd.Series,
        title: str = "Returns Distribution",
        figsize: Tuple[int, int] = (10, 6)
    ) -> plt.Figure:
        """
        Plot distribution of returns.

        Args:
            returns: Series of returns
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        ax.hist(returns.dropna(), bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(returns.mean(), color='red', linestyle='--',
                   label=f'Mean: {returns.mean():.4f}', linewidth=2)
        ax.axvline(returns.median(), color='green', linestyle='--',
                   label=f'Median: {returns.median():.4f}', linewidth=2)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Returns', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_simulation_results(
        result: SimulationResult,
        num_paths_to_plot: int = 100,
        figsize: Tuple[int, int] = (14, 8)
    ) -> plt.Figure:
        """
        Plot Monte Carlo simulation results.

        Args:
            result: SimulationResult object
            num_paths_to_plot: Number of simulation paths to display
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)

        # Plot simulation paths
        n_paths = min(num_paths_to_plot, result.num_simulations)
        for i in range(n_paths):
            ax1.plot(
                result.simulated_paths.iloc[:, i],
                alpha=0.1,
                color='blue',
                linewidth=0.5
            )

        # Plot mean path
        mean_path = result.simulated_paths.mean(axis=1)
        ax1.plot(mean_path, color='red', linewidth=2, label='Mean Path')

        # Plot percentiles
        p5 = result.simulated_paths.quantile(0.05, axis=1)
        p95 = result.simulated_paths.quantile(0.95, axis=1)
        ax1.fill_between(
            range(len(mean_path)),
            p5,
            p95,
            alpha=0.3,
            color='gray',
            label='5th-95th Percentile'
        )

        ax1.set_title(
            f'Monte Carlo Simulation: {result.portfolio_name} ({result.num_simulations:,} simulations)',
            fontsize=14,
            fontweight='bold'
        )
        ax1.set_xlabel('Days', fontsize=12)
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot final value distribution
        ax2.hist(result.final_values, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(
            result.statistics['mean'],
            color='red',
            linestyle='--',
            label=f"Mean: ${result.statistics['mean']:,.2f}",
            linewidth=2
        )
        ax2.axvline(
            result.statistics['var_95'],
            color='orange',
            linestyle='--',
            label=f"VaR 95%: ${result.statistics['var_95']:,.2f}",
            linewidth=2
        )

        ax2.set_title('Distribution of Final Portfolio Values', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Final Value ($)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_portfolio_allocation(
        portfolio: Portfolio,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot portfolio allocation as a pie chart.

        Args:
            portfolio: Portfolio object
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        symbols = list(portfolio.assets.keys())
        weights = [portfolio.assets[s] for s in symbols]
        values = [portfolio.get_asset_value(s) for s in symbols]

        # Pie chart for weights
        ax1.pie(
            weights,
            labels=symbols,
            autopct='%1.1f%%',
            startangle=90,
            textprops={'fontsize': 10}
        )
        ax1.set_title('Portfolio Allocation by Weight', fontsize=12, fontweight='bold')

        # Bar chart for values
        colors = plt.cm.Set3(np.linspace(0, 1, len(symbols)))
        ax2.bar(symbols, values, color=colors, edgecolor='black')
        ax2.set_title('Portfolio Allocation by Value', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Symbol', fontsize=10)
        ax2.set_ylabel('Value ($)', fontsize=10)
        ax2.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, (symbol, value) in enumerate(zip(symbols, values)):
            ax2.text(i, value, f'${value:,.0f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_correlation_matrix(
        data: pd.DataFrame,
        title: str = "Correlation Matrix",
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot correlation matrix as a heatmap.

        Args:
            data: DataFrame with returns or prices
            title: Plot title
            figsize: Figure size

        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate correlation matrix
        corr = data.corr()

        # Create heatmap
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')

        # Set ticks
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right')
        ax.set_yticklabels(corr.columns)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Correlation', rotation=270, labelpad=20)

        # Add correlation values
        for i in range(len(corr.columns)):
            for j in range(len(corr.columns)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)

        ax.set_title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        return fig

    @staticmethod
    def save_figure(fig: plt.Figure, filename: str, dpi: int = 300) -> None:
        """
        Save figure to file.

        Args:
            fig: matplotlib Figure object
            filename: Output filename
            dpi: Resolution in dots per inch
        """
        fig.savefig(filename, dpi=dpi, bbox_inches='tight')
