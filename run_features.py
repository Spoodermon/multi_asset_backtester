import os
import pandas as pd
from data_fetcher.fetcher import EquityFetcher
from features.pipeline import build_feature_pipeline

if __name__ == "__main__":
    # Ensure data exists
    fetcher = EquityFetcher(data_dir="data")
    fetcher.fetch_and_save(tickers=["AAPL", "MSFT"],
                           start="2020-01-01")
    
    pipeline = build_feature_pipeline()
    print("Pipeline steps:", pipeline.steps)
    union = pipeline.steps[0][1]  # Get the FeatureUnion step
    cols = union.get_feature_names_out()
    
    # Load, transform, and save features
    for ticker in ["AAPL", "MSFT"]:
        df = pd.read_parquet(f"data/{ticker}.parquet")
        
        # 1. Generate numpy array
        arr = pipeline.fit_transform(df)
        

        # 2. Wrap back into DataFrame
        feat_df = pd.DataFrame(arr, index=df.index, columns=cols)
        
        # 3. Save to Parquet       
        out_path = f"data/{ticker}_features.parquet"
        feat_df.to_parquet(out_path)
        print(f"saved features for {ticker} -> {out_path}")