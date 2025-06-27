import os
import pandas as pd
import numpy as np
import tensorflow as tf
import vectorbt as vbt
import matplotlib.pyplot as plt

DATA_DIR   = "data"
MODEL_PATH = "models/return_predictor.h5"
TICKERS  = ["AAPL", "MSFT", "QQQ", "GS"]

def load_features_and_prices(tickers):
    feats, prices = {}, {}
    for t in tickers:
        feats[t]  = pd.read_parquet(os.path.join(DATA_DIR, f"{t}_features.parquet"))
        prices[t] = pd.read_parquet(os.path.join(DATA_DIR, f"{t}.parquet"))["Close"]
    return feats, prices

def predict_returns(feats, model):
    # concatenate tickers along columns, but we need panel-style input
    all_preds = {}
    for t, df in feats.items():
        X = df.to_numpy()
        pred = model.predict(X).flatten()
        # align back into a Series indexed by the feature DataFrame’s index
        all_preds[t] = pd.Series(pred, index=df.index, name=t)
    return pd.DataFrame(all_preds)

if __name__ == "__main__":
    # 1) Load data
    feats, prices = load_features_and_prices(TICKERS)

    # 2) Load model
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 3) Predict
    pred_df = predict_returns(feats, model)
    print("Predictions head:\n", pred_df.head())

    # simple long/short signal: go long if pred > 0, short if pred < 0
    # you could also rank across tickers and take top/bottom n
    #signal = pred_df.applymap(lambda x: 1 if x > 0 else -1)
    signal = pred_df.gt(0).astype(int).replace({0: -1})
    
    # 4) Fetch price series as a DataFrame
    price_df = pd.concat(prices, axis=1)
    #yields a Dataframe with one column per ticker indexed by timestamp index of series
    
    # 5) Build and run the backtest
    pf = vbt.Portfolio.from_signals(
        close=price_df,
        entries=signal == 1,
        exits =signal == -1,
        freq='1D',
        init_cash=1e6,
        fees=0.0005
    )

    # 6) Inspect results
    print("Total return:", pf.total_return())
    print("Sharpe ratio:", pf.sharpe_ratio())
    
    value_df = pf.value()           # DataFrame with one column per ticker
    ax = value_df.plot(
        title="Portfolio Value Over Time",
        xlabel="Date",
        ylabel="Portfolio Value"
    )
    ax.legend(title="Ticker")
    plt.tight_layout()
    plt.show()
    
    # Plot the order history for the "AAPL" ticker
    #pf["AAPL"].orders.plot()

