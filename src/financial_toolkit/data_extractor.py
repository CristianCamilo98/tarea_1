"""
Data Extractor Module

Provides functionality to fetch financial data from multiple APIs.
"""

from typing import List, Optional
from datetime import datetime
import pandas as pd
from .yahoo_finance import YahooFinanceExtractor
from .alpha_vantage import AlphaVantageExtractor
import os


class DataExtractor:
    """
    Multi-API data extractor for financial market data.

    Supports multiple data sources including Yahoo Finance and Alpha Vantage.
    """

    def __init__(self, alpha_vantage_key: Optional[str] = None):
        """
        Initialize the data extractor.

        Args:
            alpha_vantage_key: API key for Alpha Vantage (optional, can be set via environment variable ALPHA_VANTAGE_API_KEY)
        """
        if alpha_vantage_key is None:
          alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY', None)
        self.alpha_vantage_key = alpha_vantage_key
        self.available_sources = ['yahoo']
        if alpha_vantage_key:
            self.available_sources.append('alpha_vantage')
            self.alpha_vantage_extractor = AlphaVantageExtractor(api_key=alpha_vantage_key)
        self.yahoo_finance_extractor = YahooFinanceExtractor()


    def fetch_historic_prices(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo'
    ) -> pd.DataFrame:
        """
        Fetch historical prices for multiple symbols and return as a DataFrame.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            source: Data source

        Returns:
            DataFrame with multi-index (date, symbol)
        """
        all_data = []

        for symbol in symbols:
            try:
                if source == 'yahoo':
                    price_data_list = self.yahoo_finance_extractor.fetch_historical_prices(symbol, start_date, end_date)
                elif source == 'alpha_vantage':
                    price_data_list = self.alpha_vantage_extractor.fetch_historical_prices(symbol, start_date, end_date)
                else:
                    raise ValueError(f"Unsupported source: {source}")

                for price_data in price_data_list:
                    all_data.append(price_data.to_dict())
            except Exception as e:
                print(f"Warning: Failed to fetch data for {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No data fetched for any symbol")

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'symbol']).sort_index()

        return df
