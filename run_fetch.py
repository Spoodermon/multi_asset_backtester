from data_fetcher.fetcher import EquityFetcher

#Guards against running this script directly
if __name__ == "__main__":
    fetcher = EquityFetcher(data_dir="data")
    fetcher.fetch_and_save(tickers = ["AAPL", "SPY", "QQQ"],
                           start="2020-01-01"
                           )
