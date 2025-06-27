import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

# 1. Parameters
FEATURE_DIR = "data"
TICKERS  = ["AAPL", "MSFT", "QQQ", "GS"]
TEST_SIZE = 0.2
VAL_SIZE = 0.1
RANDOM_SEED = 42 #HHG2U

# 2. Load features and compute labels
def load_data(tickers):
    dfs = []
    for t in tickers:
        feat   = pd.read_parquet(f"data/{t}_features.parquet")
        raw_df = pd.read_parquet(f"data/{t}.parquet")
        price  = raw_df["Close"] if isinstance(raw_df, pd.DataFrame) else raw_df

        # compute next-day return
        ret = price.pct_change().shift(-1)

        # ----- COERCE TO 1-D SERIES -----
        if isinstance(ret, pd.DataFrame):
            # if it is a DataFrame with exactly one column, extract that column
            if ret.shape[1] == 1:
                ret = ret.iloc[:, 0]
            else:
                raise ValueError(f"Unexpected multiple columns in return DataFrame for {t}")

        # now ret is a pd.Series, so .to_frame works perfectly
        ret_df = ret.to_frame(name="target")

        df = feat.join(ret_df, how="inner").dropna()
        dfs.append(df)

    all_data = pd.concat(dfs, axis=0)
    # debug: print(all_data.columns.tolist())

    # — optional sanity check —  
    print("⚙️ all_data.columns =", all_data.columns.tolist())
    
    arr = all_data.to_numpy()
    X   = arr[:, :-1]
    y   = arr[:,  -1]
    return X, y


    

# 3. Split data into train, validation, and test sets
X, y = load_data(TICKERS)
X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, 
                                                  test_size=TEST_SIZE+VAL_SIZE,
                                                  random_state=RANDOM_SEED)
val_fraction = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, 
                                                  test_size=val_fraction,
                                                  random_state=RANDOM_SEED)
print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

def build_model(input_dim):
    """Simple feedforward neural network for regression.

    Dense -> ReLU -> Dropout -> Dense -> Linear output
    """
    inputs = keras.Input(shape=(input_dim,), name="features")
    x = keras.layers.Dense(64, activation='relu')(inputs) #small network guards against overfitting on limited data
    x = keras.layers.Dropout(0.2)(x) #intoduces regularization
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(1, activation="linear", name="return")(x)  # Linear output for regression
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="return_predictor")
    model.compile(optimizer="adam", # Adam optimizer
                  loss="mse", # Mean Squared Error for regression
                  metrics=[keras.metrics.RootMeanSquaredError()]) # RMSE as metric
    return model #MSE loss + RMSE metric aligns with regression on returns

# 4. Instnatiate
model = build_model(X_train.shape[1])
model.summary()  # Print model architecture


# 5. Callbacks
logdir = os.path.join("logs", "fit")
tensorboard_cb = keras.callbacks.TensorBoard(log_dir=logdir) #allows visualization of training progress in TensorBoard
earlystop_cb = keras.callbacks.EarlyStopping(monitor='val_loss',
                                             patience=5,
                                             restore_best_weights=True) #prevents over-training and rolls back to best epoch

# 6. Fit
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,  # Adjust based on convergence
                    batch_size=256,  # Adjust based on memory constraints
                    callbacks=[tensorboard_cb, earlystop_cb])

# 7. Save model and history
model.save("models/return_predictor.h5")
pd.DataFrame(history.history).to_csv("models/training_history.csv", index=False)
print("Model and training history saved.")