"""
Data Extractor Module

Provides functionality to fetch financial data from multiple APIs.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import pandas as pd
import requests
from .data_models import PriceData
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
          alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.alpha_vantage_key = alpha_vantage_key
        self.available_sources = ['yahoo']
        if alpha_vantage_key:
            self.available_sources.append('alpha_vantage')

    def fetch_yahoo_finance(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PriceData]:
        """
        Fetch data from Yahoo Finance using yfinance library.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data fetch
            end_date: End date for data fetch

        Returns:
            List of PriceData objects
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance package is required. Install with: pip install yfinance")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError(f"No data found for symbol {symbol}")

        price_data_list = []
        for date, row in df.iterrows():
            price_data = PriceData(
                symbol=symbol.upper(),
                date=date.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
                adjusted_close=float(row['Close']),
                source='yahoo'
            )
            price_data_list.append(price_data)

        return price_data_list

    def fetch_alpha_vantage(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> List[PriceData]:
        """
        Fetch data from Alpha Vantage API.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data fetch (not used by Alpha Vantage API)
            end_date: End date for data fetch (not used by Alpha Vantage API)

        Returns:
            List of PriceData objects
        """
        if not self.alpha_vantage_key:
            raise ValueError("Alpha Vantage API key is required")

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.alpha_vantage_key,
            "outputsize": "full"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            raise ValueError(f"No data found for symbol {symbol}")

        price_data_list = []
        for date_str, values in time_series.items():
            date = datetime.strptime(date_str, "%Y-%m-%d")

            if start_date and date < start_date:
                continue
            if end_date and date > end_date:
                continue

            price_data = PriceData(
                symbol=symbol.upper(),
                date=date,
                open=float(values["1. open"]),
                high=float(values["2. high"]),
                low=float(values["3. low"]),
                close=float(values["4. close"]),
                volume=int(values["5. volume"]),
                adjusted_close=float(values["4. close"]),
                source='alpha_vantage'
            )
            price_data_list.append(price_data)

        return sorted(price_data_list, key=lambda x: x.date)

    def fetch_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo'
    ) -> List[PriceData]:
        """
        Fetch data from the specified source.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date for data fetch
            end_date: End date for data fetch
            source: Data source ('yahoo' or 'alpha_vantage')

        Returns:
            List of PriceData objects
        """
        if source not in self.available_sources:
            raise ValueError(f"Source {source} not available. Available: {self.available_sources}")

        if source == 'yahoo':
            return self.fetch_yahoo_finance(symbol, start_date, end_date)
        elif source == 'alpha_vantage':
            return self.fetch_alpha_vantage(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported source: {source}")

    def fetch_multiple(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        source: str = 'yahoo'
    ) -> pd.DataFrame:
        """
        Fetch data for multiple symbols.

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
                price_data_list = self.fetch_data(symbol, start_date, end_date, source)
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
