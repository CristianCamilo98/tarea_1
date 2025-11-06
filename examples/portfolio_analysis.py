"""
Example: Data Extraction and Cleaning

This demonstrates how to fetch, clean, and prepare data for analysis.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.financial_toolkit import DataExtractor
from src.financial_toolkit import Portfolio
from dotenv import load_dotenv
from pprint import pprint
load_dotenv()


def run_data_extraction_demo(source, symbols):
    print("=" * 70)
    print("Data Extraction of Financial Market Data Demo using source:" f" {source}")
    print("=" * 70)

    # 1. Initialize extractor
    print("1. Initializing data extractor...")
    extractor = DataExtractor()

    # 2.1 Fetch real data using yahoo finance
    start_date = datetime.now() - timedelta(days=365*10)
    end_date = datetime.now()

    print(f"\n2.1 Fetch data using yahoo finance for {symbols}...")
    data = extractor.fetch_historic_prices(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source=source
    )
    print(f"   ✓ Successfully fetched real market data")

    for symbol_data in data:
        print("##################################################")
        print("####### Sample data inspection for "f"{symbol_data.prices[0].symbol} ##########")
        print("##################################################")
        data_df = symbol_data.to_dataframe()
        print(f"  ✓ Sample shape for {symbol_data.prices[0].symbol}: {data_df.shape}")
        print(f"  ✓ Sample data for {symbol_data.prices[0].symbol} (first row):")
        print(data_df.head(1).to_string())
        print("\n\n")
        print(" Statistics summary:")
        pprint(symbol_data.get_statistics_summary())
        print("\n\n")
        print("Percentile 95%:")
        percentile_95 = symbol_data.get_price_percentile(0.95)
        print(percentile_95)
        print("Get moving average (window=3), sample first 5 values:")
        moving_avg = symbol_data.get_moving_average(window=3)
        print(moving_avg.head(5))

        print("\nAdding more historical data by fetching 5 more years...")
        end_date_new =  start_date - timedelta(days=1)
        start_date_new =  start_date - timedelta(days=365*5)

        new_data = extractor.fetch_historic_prices(
            symbols=[symbol_data.prices[0].symbol],
            start_date=start_date_new,
            end_date=end_date_new,
            source=source
        )
        for price_series_data in new_data:
            print("Type of price_series_data:", type(price_series_data))
            symbol_data.add_prices(price_series_data.prices)

        print(f"After adding more data, new shape for {symbol_data.prices[0].symbol}: {symbol_data.to_dataframe().shape}")
        print("\n\n")
        pprint(f"New statistics summary:")
        pprint(symbol_data.get_statistics_summary())
        print("\n\n")
        print("Get moving average (window=3) and printing first 5 values:")
        moving_avg = symbol_data.get_moving_average(window=3)
        print(moving_avg.head(5))
        print("\n\n")

    print("="*70)
    splits = extractor.fetch_splits(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        source=source
    )
    print(f"   ✓ Successfully fetched stock splits data")
    for split_data in splits:
        print(f"Symbol: {split_data.symbol}")
        if split_data.data.empty:
            print("  No splits data available.")
        else:
            print(f"Splits sample data:")
            print(split_data.data.head(5))

    print("="*70)
    print("CREATING A PORTFOLIO WITH THE FETCHED DATA")
    portfolio_info = {}
    for symbol_data in data:
        # Define weight so it sums to 1.0 so outside of the loop we can create the portfolio
        weight = 1.0 / len(data)
        portfolio_info[symbol_data.prices[0].symbol] = {'weight': weight, 'historical_data': symbol_data}

    portfolio = Portfolio(
        holdings={symbol: info['historical_data'] for symbol, info in portfolio_info.items()},
        weights={symbol: info['weight'] for symbol, info in portfolio_info.items()},
    )
    print(portfolio)
    monte_carlo_simulation = portfolio.monte_carlo_simulation(num_simulations=1000, horizon_days=252*2)
    monte_carlo_simulation.plots_report()
    monte_carlo_simulation.plots_detailed_report()

    print("\n" + "=" * 70)
    print("Fetching fundamental data for the symbols...")
    fundamental_data = extractor.fetch_fundamental_data(symbols=symbols, source=source)
    for symbol, data in fundamental_data.items():
        print(f"Fundamental data for {symbol}:")
        pprint(data.to_dict())
        print("\n\n")

    print("=" * 70)
    print("Generating portfolio report...")
    report = portfolio.report(monte_carlo_simulation, save_to_file=True)
    print(report)

if __name__ == '__main__':
    # symbols = ['AAPL', 'VOO']
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    # symbols = ['JNJ', 'KO', 'XOM', 'GLD', 'NEE']
    # symbols = ['NMAX', 'SRPT', 'GLOB', 'CE', 'BHVN', 'PACS', 'ENPH', 'FRPT']
    run_data_extraction_demo(source='yahoo', symbols=symbols)
    # run_data_extraction_demo(source='alpha_vantage', symbols=symbols)
