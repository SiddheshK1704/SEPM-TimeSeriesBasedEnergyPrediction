# ================= RNN MODEL FILE =================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import SimpleRNN, Dense
from tensorflow.keras.callbacks import EarlyStopping

# ---------------- SETTINGS ----------------
TRAIN_MODEL = True   # Set False to load saved model

# ---------------- CREATE FOLDERS ----------------
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# ---------------- LOAD DATA ----------------
files = [
    "data/USA_GA_Albany-Dougherty.County.AP.722160_TMY3_LOW.csv",
    "data/USA_GA_Albany-Dougherty.County.AP.722160_TMY3_BASE.csv",
    "data/USA_GA_Albany-Dougherty.County.AP.722160_TMY3_HIGH.csv"
]

dfs = []

for i, file in enumerate(files):
    df_temp = pd.read_csv(file)

    year = 2023 + i
    df_temp["Date/Time"] = f"{year} " + df_temp["Date/Time"]

    mask_24 = df_temp["Date/Time"].str.contains("24:00:00")
    df_temp.loc[mask_24, "Date/Time"] = (
        df_temp.loc[mask_24, "Date/Time"]
        .str.replace("24:00:00", "00:00:00", regex=False)
    )

    df_temp["Date/Time"] = pd.to_datetime(
        df_temp["Date/Time"], format="%Y %m/%d  %H:%M:%S"
    )

    df_temp.loc[mask_24, "Date/Time"] += pd.Timedelta(days=1)

    dfs.append(df_temp)

# ---------------- COMBINE ----------------
df = pd.concat(dfs)
df = df.sort_values("Date/Time").set_index("Date/Time")

# ---------------- TARGET ----------------
target_col = "Electricity:Facility [kW](Hourly)"
series = df[target_col]

print("Null values:", series.isnull().sum())

# ---------------- TRAIN TEST SPLIT ----------------
split = int(len(series) * 0.8)
train = series[:split]
test = series[split:]

# ---------------- SCALING ----------------
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
test_scaled = scaler.transform(test.values.reshape(-1, 1))

# ---------------- SEQUENCE CREATION ----------------
TIME_STEPS = 168  # Weekly pattern

def create_sequences(data, steps):
    X, y = [], []
    for i in range(len(data) - steps):
        X.append(data[i:i + steps])
        y.append(data[i + steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
X_test, y_test = create_sequences(test_scaled, TIME_STEPS)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

# ---------------- MODEL ----------------
model_path = "models/rnn_model.h5"

if TRAIN_MODEL or not os.path.exists(model_path):

    print("\nTraining RNN Model...")

    model = Sequential([
        SimpleRNN(64, return_sequences=True, input_shape=(TIME_STEPS, 1)),
        SimpleRNN(32),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")

    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=7,
        restore_best_weights=True
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    # ---------------- SAVE MODEL ----------------
    model.save(model_path)
    print("✅ RNN model saved")

else:
    print("\nLoading saved RNN model...")
    model = load_model(model_path)
    history = None

# ---------------- PREDICTION ----------------
y_pred = model.predict(X_test)

# ---------------- INVERSE SCALING ----------------
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

# ---------------- METRICS ----------------
mse = mean_squared_error(y_test_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n📊 RNN RESULTS")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# ---------------- SAVE OUTPUT ----------------
np.save("outputs/rnn_pred.npy", y_pred_inv)
np.save("outputs/rnn_test.npy", y_test_inv)

# ---------------- PLOT ----------------
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.legend()
plt.title("RNN Forecast")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("outputs/rnn_result.png")
plt.close()

# ---------------- LOSS CURVE ----------------
if history is not None:
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("RNN Training vs Validation Loss")
    plt.savefig("outputs/rnn_loss.png")
    plt.close()

print("\n✅ RNN MODEL DONE")