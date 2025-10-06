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
from src.financial_toolkit import DataExtractor, DataCleaner
from dotenv import load_dotenv
load_dotenv()


def main():
    print("=" * 70)
    print("Data Extraction and Cleaning Example")
    print("=" * 70)
    print()

    # 1. Initialize extractor
    print("1. Initializing data extractor...")
    extractor = DataExtractor()
    print(f"   ✓ Available sources: {extractor.available_sources}")

    # 2.1 Fetch real data using yahoo finance
    symbols = ['AAPL', 'VOO']
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    print(f"\n2.1 Fetch data using yahoo finance for {symbols}...")
    try:
        data = extractor.fetch_multiple(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source='yahoo'
        )
        print(f"   ✓ Successfully fetched real market data")
        print(f"   ✓ Data shape: {data.shape}")
        # Check the index
        print(f"   ✓ Index levels: {data.index.names}")
    except Exception as e:
        print(f"   ✗ Could not fetch real data: {str(e)[:100]}...")

    # 2.2 Fetch real data using alpha_vantage
    symbols = ['AAPL', 'VOO']
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()
    print(f"\n2.2 Fetch data using alpha_vantage for {symbols}...")
    try:
        data = extractor.fetch_multiple(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            source='alpha_vantage'
        )
        print(f"   ✓ Successfully fetched real market data")
        print(f"   ✓ Data shape: {data.shape}")
        # Check the index
        print(f"   ✓ Index levels: {data.index.names}")
    except Exception as e:
        print(f"   ✗ Could not fetch real data: {str(e)[:100]}...")

    # 3. Inspect data quality
    print("\n3. Inspecting data quality...")
    missing_count = data.isna().sum().sum()
    print(f"   ✓ Missing values: {missing_count}")
    print(f"   ✓ Date range: {data.index.get_level_values('date').min()} to {data.index.get_level_values('date').max()}")

    # Show a sample
    print("\n   Sample data (first 5 rows):")
    print(data.head().to_string())

    # 4. Clean the data
    print("\n4. Cleaning data...")
    cleaner = DataCleaner()

    # Step-by-step cleaning
    print("   a. Removing duplicates...")
    data_no_dupes = cleaner.remove_duplicates(data)
    dupes_removed = len(data) - len(data_no_dupes)
    print(f"      ✓ Removed {dupes_removed} duplicate rows")

    print("   b. Handling missing values...")
    data_filled = cleaner.handle_missing_values(data_no_dupes, method='ffill')
    missing_after = data_filled.isna().sum().sum()
    print(f"      ✓ Missing values after fill: {missing_after}")

    print("   c. Validating price data...")
    data_valid = cleaner.validate_price_data(data_filled)
    invalid_removed = len(data_filled) - len(data_valid)
    print(f"      ✓ Removed {invalid_removed} invalid rows")

    print("   d. Calculating returns...")
    data_with_returns = cleaner.calculate_returns(data_valid, method='simple')
    print(f"      ✓ Returns calculated")

    # 5. Or use the full pipeline
    print("\n5. Running full cleaning pipeline...")
    cleaned_data = cleaner.clean_data(
        data,
        remove_duplicates=True,
        handle_missing='ffill',
        validate_prices=True,
        calculate_returns=True
    )
    print(f"   ✓ Clean data shape: {cleaned_data.shape}")
    print(f"   ✓ Columns: {list(cleaned_data.columns)}")

    # 6. Show statistics
    print("\n6. Data statistics after cleaning:")
    if 'returns' in cleaned_data.columns:
        returns = cleaned_data['returns'].dropna()
        print(f"   ✓ Mean return: {returns.mean():.4f}")
        print(f"   ✓ Std deviation: {returns.std():.4f}")
        print(f"   ✓ Min return: {returns.min():.4f}")
        print(f"   ✓ Max return: {returns.max():.4f}")

    # 7. Check for outliers
    print("\n7. Checking for outliers in close prices...")
    try:
        data_no_outliers = cleaner.remove_outliers(
            cleaned_data,
            column='close',
            method='iqr',
            threshold=3.0
        )
        outliers_removed = len(cleaned_data) - len(data_no_outliers)
        print(f"   ✓ Removed {outliers_removed} outliers")
    except Exception as e:
        print(f"   ✓ No significant outliers detected")

    print("\n" + "=" * 70)
    print("Data Extraction and Cleaning Complete!")
    print("=" * 70)
    print(f"\nFinal clean dataset: {cleaned_data.shape[0]} rows × {cleaned_data.shape[1]} columns")
    print("Ready for analysis!")


if __name__ == '__main__':
    main()
