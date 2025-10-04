"""
Example: Complete Financial Analysis Workflow

This example demonstrates the full capabilities of the financial toolkit.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from src.financial_toolkit import (
    DataExtractor,
    DataCleaner,
    Portfolio,
    PortfolioSimulator,
    ReportGenerator,
    Visualizer
)


def main():
    """Run a complete financial analysis workflow."""
    print("=" * 80)
    print("Financial Market Data Toolkit - Complete Example")
    print("=" * 80)
    
    # Step 1: Extract Data
    print("\n1. Extracting market data...")
    extractor = DataExtractor()
    
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    
    try:
        data = extractor.fetch_multiple(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source='yahoo'
        )
        print(f"   ✓ Fetched data for {len(symbols)} symbols")
        print(f"   ✓ Data shape: {data.shape}")
    except Exception as e:
        print(f"   ✗ Error fetching data: {e}")
        print("   Using simulated data instead...")
        # Create simulated data for demonstration
        import pandas as pd
        import numpy as np
        dates = pd.date_range(start_date, end_date, freq='D')
        data_list = []
        for date in dates:
            for symbol in symbols:
                base_price = {'AAPL': 150, 'GOOGL': 120, 'MSFT': 300}[symbol]
                price = base_price * (1 + np.random.normal(0, 0.02))
                data_list.append({
                    'symbol': symbol,
                    'open': price * 0.99,
                    'high': price * 1.02,
                    'low': price * 0.98,
                    'close': price,
                    'volume': int(1000000 * (1 + np.random.random())),
                    'adjusted_close': price,
                    'source': 'simulated'
                })
        data = pd.DataFrame(data_list)
        data['date'] = pd.concat([pd.Series(dates)] * len(symbols)).reset_index(drop=True)
        data = data.set_index(['date', 'symbol']).sort_index()
        print(f"   ✓ Generated simulated data: {data.shape}")
    
    # Step 2: Clean Data
    print("\n2. Cleaning data...")
    cleaner = DataCleaner()
    
    cleaned_data = cleaner.clean_data(
        data,
        remove_duplicates=True,
        handle_missing='ffill',
        validate_prices=True,
        calculate_returns=True
    )
    print(f"   ✓ Data cleaned: {cleaned_data.shape}")
    print(f"   ✓ Returns calculated")
    
    # Step 3: Create Portfolio
    print("\n3. Creating portfolio...")
    portfolio = Portfolio(
        name='Diversified Tech Portfolio',
        assets={
            'AAPL': 0.4,
            'GOOGL': 0.35,
            'MSFT': 0.25
        },
        initial_value=100000.0,
        data=cleaned_data
    )
    print(f"   ✓ Portfolio created: {portfolio.name}")
    print(f"   ✓ Initial value: ${portfolio.initial_value:,.2f}")
    
    # Step 4: Run Monte Carlo Simulation
    print("\n4. Running Monte Carlo simulation...")
    simulator = PortfolioSimulator(portfolio)
    
    simulation_result = simulator.simulate(
        num_simulations=1000,
        time_horizon=252,  # 1 year
        random_seed=42
    )
    print(f"   ✓ Completed {simulation_result.num_simulations:,} simulations")
    print(f"   ✓ Time horizon: {simulation_result.time_horizon} days")
    print(f"   ✓ Expected return: {simulation_result.statistics['expected_return']:.2%}")
    print(f"   ✓ Probability of loss: {simulation_result.statistics['probability_of_loss']:.2%}")
    
    # Step 5: Calculate Risk Metrics
    print("\n5. Calculating risk metrics...")
    risk_metrics = simulator.calculate_risk_metrics()
    print(f"   ✓ Volatility: {risk_metrics['volatility']:.2%}")
    print(f"   ✓ Sharpe Ratio: {risk_metrics['sharpe_ratio']:.4f}")
    print(f"   ✓ Max Drawdown: {risk_metrics['max_drawdown']:.2%}")
    
    # Step 6: Generate Reports
    print("\n6. Generating reports...")
    report_gen = ReportGenerator()
    
    full_report = report_gen.generate_full_report(
        portfolio=portfolio,
        simulation_result=simulation_result,
        risk_metrics=risk_metrics,
        data_summary=cleaned_data
    )
    
    report_filename = 'financial_analysis_report.md'
    report_gen.save_report(full_report, report_filename)
    print(f"   ✓ Report saved to: {report_filename}")
    
    # Step 7: Create Visualizations
    print("\n7. Creating visualizations...")
    visualizer = Visualizer()
    
    # Portfolio allocation
    try:
        fig1 = visualizer.plot_portfolio_allocation(portfolio)
        visualizer.save_figure(fig1, 'portfolio_allocation.png')
        print(f"   ✓ Portfolio allocation chart saved")
    except Exception as e:
        print(f"   ✗ Could not create allocation chart: {e}")
    
    # Simulation results
    try:
        fig2 = visualizer.plot_simulation_results(simulation_result, num_paths_to_plot=50)
        visualizer.save_figure(fig2, 'simulation_results.png')
        print(f"   ✓ Simulation results chart saved")
    except Exception as e:
        print(f"   ✗ Could not create simulation chart: {e}")
    
    # Price history
    try:
        fig3 = visualizer.plot_price_history(cleaned_data, symbol='AAPL')
        visualizer.save_figure(fig3, 'price_history_aapl.png')
        print(f"   ✓ Price history chart saved")
    except Exception as e:
        print(f"   ✗ Could not create price history chart: {e}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)
    print(f"\nGenerated files:")
    print(f"  - {report_filename}")
    print(f"  - portfolio_allocation.png")
    print(f"  - simulation_results.png")
    print(f"  - price_history_aapl.png")
    print("\n")


if __name__ == '__main__':
    main()
