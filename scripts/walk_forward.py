import os
import numpy as np
import pandas as pd
import tensorflow as tf
import vectorbt as vbt
from models.train import build_model  # model factory from models/train.py
import matplotlib.pyplot as plt

# 1. Parameters
DATA_DIR = "data"
TICKERS  = ["AAPL", "MSFT", "QQQ", "GS"]
# define rolling windows: (train_start, train_end, test_start, test_end)
WINDOWS = [
    ("2020-01-01","2021-12-31","2022-01-01","2022-06-30"),
    ("2020-07-01","2022-06-30","2022-07-01","2023-01-31"),
    ("2021-01-01","2022-12-31","2023-01-01","2023-06-30"),
    ("2021-01-01","2022-12-31","2023-01-01","2024-12-31"),
    # …add as many as you like…
]

results = []

# 2. Pre-load all features & prices
feats, prices = {}, {}
for t in TICKERS:
    feats[t]  = pd.read_parquet(os.path.join(DATA_DIR, f"{t}_features.parquet"))
    prices[t] = pd.read_parquet(os.path.join(DATA_DIR, f"{t}.parquet"))["Close"]

# 3. Walk-forward loop
for train_start, train_end, test_start, test_end in WINDOWS:
    # --- a) Build training X,y ---
    X_tr_parts, y_tr_parts = [], []
    for t in TICKERS:
        Xf = feats[t].loc[train_start:train_end]
        pr = prices[t].loc[train_start:train_end]
        ret = pr.pct_change().shift(-1)
        df  = Xf.join(ret, how="inner").dropna()

        arr = df.to_numpy()
        # all cols except last → features; last col → label
        X_tr_parts.append(arr[:, :-1])
        y_tr_parts.append(arr[:,  -1])


    X_train = np.vstack(X_tr_parts)
    y_train = np.concatenate(y_tr_parts)

    # --- b) Train a fresh model ---
    model = build_model(X_train.shape[1])
    model.fit(X_train, y_train,
              epochs=20,       # adjust
              batch_size=256,
              verbose=0)

    # --- c) Predict on test ---
    pred_dict = {}
    for t in TICKERS:
        Xf_test = feats[t].loc[test_start:test_end]
        preds   = model.predict(Xf_test.to_numpy()).flatten()
        pred_dict[t] = pd.Series(preds, index=Xf_test.index)
    pred_df = pd.DataFrame(pred_dict)
    
    # Apply threshold
    # threshold = 0.005
    # signal = pred_df.gt(threshold).astype(int).replace({0: -1})
    # price_df = pd.concat(prices, axis=1).loc[test_start:test_end]


    # # --- d) Build signals & price DataFrame ---
    signal   = pred_df.gt(0).astype(int).replace({0: -1})
    price_df = pd.concat(prices, axis=1).loc[test_start:test_end]

    # --- e) Backtest via VectorBT ---
    pf = vbt.Portfolio.from_signals(
        close=price_df,
        entries=signal == 1,
        exits=  signal == -1,
        init_cash=1e6,
        fees=0.0005,
        freq='1D'
    )

    # --- f) Record metrics per ticker ---
    tr = pf.total_return()
    sr = pf.sharpe_ratio()
    for t in TICKERS:
        results.append({
            "train_start":  train_start,
            "train_end":    train_end,
            "test_start":   test_start,
            "test_end":     test_end,
            "ticker":       t,
            # pull the scalar at (t, t)
            "total_return": tr.at[t, t],
            "sharpe":       sr.at[t, t]
        })

# 4. Summarize
res_df = pd.DataFrame(results)
res_df['test_end'] = pd.to_datetime(res_df['test_end'])
print(res_df)


# 1) Boxplot of Sharpe by Ticker
sharpe_pivot = res_df.pivot(
    index='test_end',    # or any other column that uniquely identifies each fold
    columns='ticker',
    values='sharpe'
)

# Use pandas’ .plot.box() which deals with shapes internally
ax = sharpe_pivot.plot.box(
    title='Sharpe Ratio Distribution by Ticker',
    ylabel='Sharpe Ratio'
)
ax.set_xlabel('')  # no x-label needed for a boxplot
plt.tight_layout()
plt.show()

# 2) Grouped bar chart of Total Return by test_end
pivot_ret = res_df.pivot(index='test_end', columns='ticker', values='total_return')
fig, ax = plt.subplots()
pivot_ret.plot(kind='bar', ax=ax)
ax.set_title('Total Return by Test Period and Ticker')
ax.set_xlabel('Test End Date')
ax.set_ylabel('Total Return')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3) Scatter plot Sharpe vs Total Return
fig, ax = plt.subplots()
ax.scatter(res_df['sharpe'], res_df['total_return'])
for _, row in res_df.iterrows():
    ax.annotate(row['ticker'], (row['sharpe'], row['total_return']),
                textcoords="offset points", xytext=(3, 3), ha='left')
ax.set_title('Sharpe Ratio vs Total Return (All Folds)')
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Total Return')
plt.tight_layout()
plt.show()