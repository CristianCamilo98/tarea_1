"""
Yahoo Finance data extractor

Provides functionality to fetch financial data from Yahoo Finance using yfinance library.
"""

from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict
from .data_models import PriceData, DividendData, SplitsData
import yfinance as yf
import pandas as pd
import numpy as np

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

    def fetch_historical_prices_batch(
        self,
        symbols: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, List[PriceData]]:
        """
        Fetch historical prices for multiple symbols.

        Args:
            symbols: List of stock ticker symbols
            start_date: Start date for data fetch
            end_date: End date for data fetch
        Returns:
            Dictionary mapping symbols to lists of PriceData objects
        """
        # Defaults: last 365 days, in UTC
        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Make bounds UTC-aware
        start_utc = to_utc_aware(start_date)
        end_utc = to_utc_aware(end_date)


        df = yf.download(symbols, start=start_utc, end=end_utc, threads=True, progress=False, auto_adjust=True)
        fetched_symbols = df.columns.get_level_values('Ticker').unique().tolist()

        idx = df.index
        if getattr(idx, "tz", None) is None:
            df.index = idx.tz_localize("UTC")
        else:
            df.index = idx.tz_convert("UTC")


        if df.empty:
            raise ValueError(f"No data found for symbol {symbols}")

        price_data: Dict[str, List[PriceData]] = {}

        for date, row in df.iterrows():
            for symb in fetched_symbols:
                if symb not in price_data:
                    price_data[symb] = []
                open_price = row[('Open', symb)]
                high_price = row[('High', symb)]
                low_price = row[('Low', symb)]
                close_price = row[('Close', symb)]
                volume = row[('Volume', symb)]


                if np.isnan(open_price) or np.isnan(high_price) or np.isnan(low_price) or np.isnan(close_price) or np.isnan(volume):
                    continue

                price_data[symb].append(
                    PriceData(
                        symbol=symb.upper(),
                        date=date.to_pydatetime(),
                        open=float(open_price),
                        high=float(high_price),
                        low=float(low_price),
                        close=float(close_price),
                        volume=int(volume),
                        adjusted_close=float(close_price),
                        source='yahoo'
                    )
                )

        return price_data

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

    def fetch_splits(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> SplitsData:
        """
        Fetch stock split data from Yahoo Finance filtered by UTC window.
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for filtering (inclusive), defaults to now - 365 days
            end_date: End date for filtering (inclusive), defaults to now
        Returns:
            SplitsData object
        """

        if start_date is None:
            start_date = datetime.now(timezone.utc) - timedelta(days=365)
        if end_date is None:
            end_date = datetime.now(timezone.utc)

        # Make bounds UTC-aware
        start_utc = to_utc_aware(start_date)
        end_utc = to_utc_aware(end_date)

        ticker = yf.Ticker(symbol)
        splits = ticker.splits
        splits = splits.tz_convert("UTC")
        mask = (splits.index >= start_utc) & (splits.index <= end_utc)
        splits = splits[mask]



        return SplitsData(
            symbol=symbol,
            data=splits,
            source="yahoo",
        )
