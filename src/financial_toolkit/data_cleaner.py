"""
Data Cleaner Module

Provides functionality to clean and preprocess financial data.
"""

from typing import List, Optional
import pandas as pd
import numpy as np
from .data_models import PriceData


class DataCleaner:
    """
    Data cleaning and preprocessing utilities for financial data.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        pass
    
    @staticmethod
    def remove_duplicates(data: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate rows from the data.
        
        Args:
            data: DataFrame with financial data
        
        Returns:
            DataFrame without duplicates
        """
        return data[~data.index.duplicated(keep='first')]
    
    @staticmethod
    def handle_missing_values(
        data: pd.DataFrame,
        method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        Handle missing values in the data.
        
        Args:
            data: DataFrame with financial data
            method: Method to handle missing values ('ffill', 'bfill', 'interpolate', 'drop')
        
        Returns:
            DataFrame with missing values handled
        """
        df = data.copy()
        
        if method == 'ffill':
            df = df.ffill()
        elif method == 'bfill':
            df = df.bfill()
        elif method == 'interpolate':
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
        elif method == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    @staticmethod
    def remove_outliers(
        data: pd.DataFrame,
        column: str,
        method: str = 'iqr',
        threshold: float = 1.5
    ) -> pd.DataFrame:
        """
        Remove outliers from the data.
        
        Args:
            data: DataFrame with financial data
            column: Column to check for outliers
            method: Method to detect outliers ('iqr' or 'zscore')
            threshold: Threshold for outlier detection
        
        Returns:
            DataFrame without outliers
        """
        df = data.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in data")
        
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        elif method == 'zscore':
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df = df[z_scores < threshold]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    @staticmethod
    def validate_price_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate price data for consistency.
        
        Args:
            data: DataFrame with financial data
        
        Returns:
            DataFrame with only valid rows
        """
        df = data.copy()
        
        # Check if required columns exist
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Validate that high >= low
        mask = df['high'] >= df['low']
        df = df[mask]
        
        # Validate that high >= open, close
        mask = (df['high'] >= df['open']) & (df['high'] >= df['close'])
        df = df[mask]
        
        # Validate that low <= open, close
        mask = (df['low'] <= df['open']) & (df['low'] <= df['close'])
        df = df[mask]
        
        # Validate that all prices are positive
        for col in required_cols:
            df = df[df[col] > 0]
        
        return df
    
    @staticmethod
    def calculate_returns(
        data: pd.DataFrame,
        column: str = 'close',
        method: str = 'simple'
    ) -> pd.DataFrame:
        """
        Calculate returns from price data.
        
        Args:
            data: DataFrame with financial data
            column: Column to calculate returns from
            method: Method to calculate returns ('simple' or 'log')
        
        Returns:
            DataFrame with returns column added
        """
        df = data.copy()
        
        if column not in df.columns:
            raise ValueError(f"Column {column} not found in data")
        
        if method == 'simple':
            df['returns'] = df[column].pct_change()
        elif method == 'log':
            df['returns'] = np.log(df[column] / df[column].shift(1))
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return df
    
    def clean_data(
        self,
        data: pd.DataFrame,
        remove_duplicates: bool = True,
        handle_missing: Optional[str] = 'ffill',
        validate_prices: bool = True,
        calculate_returns: bool = False
    ) -> pd.DataFrame:
        """
        Apply full cleaning pipeline to the data.
        
        Args:
            data: DataFrame with financial data
            remove_duplicates: Whether to remove duplicate rows
            handle_missing: Method to handle missing values (None to skip)
            validate_prices: Whether to validate price data
            calculate_returns: Whether to calculate returns
        
        Returns:
            Cleaned DataFrame
        """
        df = data.copy()
        
        if remove_duplicates:
            df = self.remove_duplicates(df)
        
        if handle_missing:
            df = self.handle_missing_values(df, method=handle_missing)
        
        if validate_prices:
            df = self.validate_price_data(df)
        
        if calculate_returns:
            df = self.calculate_returns(df)
        
        return df
