"""
Yahoo Finance data extractor

Provides functionality to fetch financial data from Yahoo Finance using yfinance library.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from .data_models import PriceData, DividendData
import yfinance as yf
import pandas as pd
import sys

def to_utc_aware(dt: datetime) -> datetime:
    """
    Convert a datetime to a UTC-aware datetime.
    Args:
        dt: Input datetime (naive or timezone-aware)
    Returns:
        UTC-aware datetime
    """
    if dt.tzinfo is None:
        # interpret naive as UTC
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

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

    def fetch_dividends(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> List[DividendData]:
        """
        Fetch dividend data from Yahoo Finance (payment dates) filtered by UTC window.
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

        # Get dividends series: index = dividend payment timestamps
        ticker = yf.Ticker(symbol)
        div = ticker.dividends
        if div.empty:
            return []

        # Normalize index to UTC (handle tz-aware vs naive indexes)
        idx = div.index
        if getattr(idx, "tz", None) is None:
            div.index = idx.tz_localize("UTC")
        else:
            div.index = idx.tz_convert("UTC")

        # Slice by UTC window (vectorized, no Python loop needed)
        filtered = div.loc[pd.Timestamp(start_utc):pd.Timestamp(end_utc)]
        if filtered.empty:
            return []

        # Build results
        out: List[DividendData] = [
            DividendData(
                symbol=symbol,
                date=ts.to_pydatetime(),
                dividend=float(amount),
                source="yahoo",
            )
            for ts, amount in filtered.items()
        ]
        return out
