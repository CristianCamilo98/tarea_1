"""
Alpha Vantage data Extractor

Provides functionality to fetch financial data from Alpha Vantage API.
"""
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from .data_models import PriceData, DividendData
import requests
from .yahoo_finance import to_utc_aware
import sys

class AlphaVantageExtractor:
    """
    Extractor for Alpha Vantage data.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def fetch_historical_prices(
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
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if response.status_code == 429:
            raise ValueError("Alpha Vantage API rate limit exceeded")

        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")

        time_series = data.get("Time Series (Daily)", {})
        if not time_series:
            raise ValueError(f"No data found for symbol {symbol}")

        if start_date is None:
            start_date = datetime.now() - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now()

        # Make bounds UTC-aware
        start_utc = to_utc_aware(start_date)
        end_utc = to_utc_aware(end_date)

        price_data_list = []
        for date_str, values in time_series.items():
            date = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            if start_utc and date < start_utc:
                continue
            if end_utc and date > end_utc:
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

    def fetch_dividends(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[DividendData]:
        """
        Fetch dividend data from Alpha Vantage (payment dates) filtered by UTC window
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for filtering (inclusive)
            end_date: End date for filtering (inclusive)
        Returns:
            List of DividendData objects
        """
        # Defaults: last 365 days, in UTC
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Make bounds UTC-aware
        start_utc = to_utc_aware(start_date)
        end_utc = to_utc_aware(end_date)

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "DIVIDENDS",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "full"
        }
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if response.status_code == 429:
            raise ValueError("Alpha Vantage API rate limit exceeded")
        if "Error Message" in data:
            raise ValueError(f"Alpha Vantage API error: {data['Error Message']}")
        dividends = data.get("data", [])
        if not dividends:
            raise ValueError(f"No dividend data found for symbol {symbol}")

        dividend_data_list: List[DividendData] = [
            DividendData(
                symbol=symbol.upper(),
                date=datetime.strptime(entry["payment_date"], "%Y-%m-%d").replace(tzinfo=timezone.utc),
                dividend=float(entry["amount"]),
                source='alpha_vantage'
            )
            for entry in dividends
            if entry["payment_date"] != 'None' and (start_utc <= datetime.strptime(entry["payment_date"], "%Y-%m-%d") <= end_utc)
        ]


        return sorted(dividend_data_list, key=lambda x: x.date)
