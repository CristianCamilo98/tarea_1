"""
Alpha Vantage data Extractor

Provides functionality to fetch financial data from Alpha Vantage API.
"""
from datetime import datetime
from typing import List, Optional
from .data_models import PriceData
import requests

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
