"""
Yahoo Finance data extractor

Provides functionality to fetch financial data from Yahoo Finance using yfinance library.
"""

from datetime import datetime, timedelta
from typing import List, Optional
from .data_models import PriceData

class YahooFinanceExtractor:
    """
    Extractor for Yahoo Finance data.
    """
    def __init__(self):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance package is required. Install with: pip install yfinance")
        self.yf = yf

    def fetch_historical_prices(
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
