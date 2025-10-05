"""
Example: Basic Usage of Financial Toolkit

This demonstrates the simplest way to use the toolkit with simulated data.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.financial_toolkit import (
    Portfolio,
    PortfolioSimulator,
    ReportGenerator,
    Visualizer
)


def create_simulated_data(symbols, days=365):
    """Create simulated market data for demonstration."""
    print(f"Creating simulated data for {len(symbols)} symbols over {days} days...")

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )

    data_list = []
    base_prices = {'AAPL': 150, 'GOOGL': 120, 'MSFT': 300, 'AMZN': 130, 'TSLA': 200}

    for date in dates:
        for symbol in symbols:
            base_price = base_prices.get(symbol, 100)
            # Simulate realistic price movements
            price = base_price * (1 + np.random.normal(0, 0.015))

            data_list.append({
                'symbol': symbol,
                'date': date,
                'open': price * 0.99,
                'high': price * 1.015,
                'low': price * 0.985,
                'close': price,
                'volume': int(1000000 * (1 + np.random.random())),
                'adjusted_close': price,
                'source': 'simulated'
            })

    df = pd.DataFrame(data_list)
    df = df.set_index(['date', 'symbol']).sort_index()

    print(f"✓ Generated {df.shape[0]} data points")
    return df


def main():
    print("=" * 70)
    print("Financial Toolkit - Basic Example")
    print("=" * 70)
    print()

    # 1. Create simulated data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    data = create_simulated_data(symbols, days=180)

    # 2. Create a portfolio
    print("\n1. Creating portfolio...")
    portfolio = Portfolio(
        name='Tech Portfolio',
        assets={
            'AAPL': 0.50,
            'GOOGL': 0.30,
            'MSFT': 0.20
        },
        initial_value=50000.0,
        data=data
    )
    print(f"   ✓ Portfolio: {portfolio.name}")
    print(f"   ✓ Initial value: ${portfolio.initial_value:,.2f}")

    # 3. Run simulation
    print("\n2. Running Monte Carlo simulation...")
    simulator = PortfolioSimulator(portfolio)
    result = simulator.simulate(
        num_simulations=500,
        time_horizon=90,  # 3 months
        random_seed=42
    )

    print(f"   ✓ {result.num_simulations} simulations completed")
    print(f"   ✓ Expected return: {result.statistics['expected_return']:.2%}")
    print(f"   ✓ Risk of loss: {result.statistics['probability_of_loss']:.2%}")

    # 4. Calculate risk metrics
    print("\n3. Calculating risk metrics...")
    risk_metrics = simulator.calculate_risk_metrics()
    print(f"   ✓ Volatility: {risk_metrics['volatility']:.2%}")
    print(f"   ✓ Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")

    # 5. Generate report
    print("\n4. Generating report...")
    report_gen = ReportGenerator()

    portfolio_report = report_gen.generate_portfolio_report(portfolio)
    print(portfolio_report)

    sim_report = report_gen.generate_simulation_report(result)
    print(sim_report)

    # 6. Create visualizations
    print("5. Creating visualizations...")
    visualizer = Visualizer()

    # Portfolio allocation
    fig1 = visualizer.plot_portfolio_allocation(portfolio)
    print("   ✓ Portfolio allocation chart created")

    # Simulation results
    fig2 = visualizer.plot_simulation_results(result, num_paths_to_plot=50)
    print("   ✓ Simulation results chart created")

    print("\n" + "=" * 70)
    print("Example Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
