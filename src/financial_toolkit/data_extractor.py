"""
Data Extractor Module

Provides functionality to fetch financial data from multiple APIs.
"""

from typing import List, Optional
from datetime import datetime
import pandas as pd
from .yahoo_finance import YahooFinanceExtractor
from .alpha_vantage import AlphaVantageExtractor
from .data_models import PriceSeriesData, SplitsData
import os
import sys


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
    ) -> list[PriceSeriesData]:
        """
        Fetch historical prices for multiple symbols and return as a DataFrame.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            source: Data source

        Returns:
            List of PriceSeriesData objects
        """
        all_data = []

        if source == 'yahoo':
            price_series = self.yahoo_finance_extractor.fetch_historical_prices_batch(symbols, start_date, end_date)
            for value in price_series.values():
                all_data.append(PriceSeriesData(prices=value))
        elif source == 'alpha_vantage':
            for symbol in symbols:
                    price_series = PriceSeriesData(prices=self.alpha_vantage_extractor.fetch_historical_prices(symbol, start_date, end_date))
                    all_data.append(price_series)
        else:
            raise ValueError(f"Unsupported source: {source}")

        if not all_data:
            raise ValueError("No data fetched for any symbol")

        return all_data

    def fetch_dividends(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo'
    ) -> pd.DataFrame:
        """
        Fetch dividend data for multiple symbols and return as a DataFrame.

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
                    dividend_data_list = self.yahoo_finance_extractor.fetch_dividends(symbol, start_date, end_date)
                elif source == 'alpha_vantage':
                    dividend_data_list = self.alpha_vantage_extractor.fetch_dividends(symbol, start_date, end_date)
                else:
                    raise ValueError(f"Unsupported source: {source}")

                for dividend_data in dividend_data_list:
                    all_data.append(dividend_data.to_dict())
            except Exception as e:
                print(f"Warning: Failed to fetch dividend data for {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No dividend data fetched for any symbol")

        df = pd.DataFrame(all_data)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index(['date', 'symbol']).sort_index()

        return df

    def fetch_splits(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo'
    ) -> list[SplitsData]:
        """
        Fetch stock split data for multiple symbols and return as a DataFrame.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
            source: Data source. Currently supports 'yahoo' and 'alpha_vantage'.
        Returns:
            DataFrame with multi-index (date, symbol)
        """
        all_data = []

        for symbol in symbols:
            try:
                if source == 'yahoo':
                    split_data = self.yahoo_finance_extractor.fetch_splits(symbol, start_date, end_date)
                elif source == 'alpha_vantage':
                    split_data = self.alpha_vantage_extractor.fetch_splits(symbol, start_date, end_date)
                else:
                    raise ValueError(f"Unsupported source: {source}")

                all_data.append(split_data)
            except Exception as e:
                print(f"Warning: Failed to fetch split data for {symbol}: {e}")
                continue

        if not all_data:
            raise ValueError("No split data fetched for any symbol")

        return all_data
