# fetcher.py: Functions for fetching data

import os
import pandas as pd
import yfinance as yf
"""
basically wrap yfinance's .download() in a class so its reusable and can be extended later

store data in Parquet for compactness and fast reloads 

Encapsulates all fetching logic in one place. 

If I get boujee and can afford annother data source, I can subclass a new 
FIxedIncomeFetcher or CryptoFetcher from this class and reuse the same logic
"""

class EquityFetcher:
    """
    Fetch historical OHCLV data for a list of equities using yfinance.
    save as Parquet files in ./data/.
    """
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir # Directory to save data files
        os.makedirs(self.data_dir, exist_ok= True) #Ensures directory exists, if not creates
        
    def fetch_and_save(self,
                       tickers: list[str],
                       start: str = "2010-01-01",
                       end: str = None,
                       interval: str = "1d") -> None:
        """
        For each ticker, download the OHCLV history and save it to 
        data/<TICKER>.parquet.
        
        :param tickers: List of ticker symbols to fetch data for.
        :param start: Start date for fetching data (YYYY-MM-DD).
        :param end: End date for fetching data (YYYY-MM-DD).
        """
        for ticker in tickers:
            print(f"Fetching {ticker}...")
            # Ensure ticker is a string and strip any whitespace
            try:
                df = yf.download(ticker, start=start, end=end, interval=interval)
                if not df.empty:
                    file_path = os.path.join(self.data_dir, f"{ticker}.parquet")
                    df.to_parquet(file_path) #Columnar storage is efficient for time-series data (fast reads for slices)
                    #as opposed to CSV, it preserves dtypes (timestamps, floats, etc.) and is compressed by default
                    print(f"Saved {ticker} to {file_path}")
                else:
                    print(f"Warning: no data for {ticker}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}") #Printing per ticker helps spot failures