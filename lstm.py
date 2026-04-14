import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

import os

# Train model- True. Load model-False.
TRAIN_MODEL = False

#Loading datasets
files = [
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_LOW.csv",
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_BASE.csv",
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_HIGH.csv"
]

dfs = []

for i, file in enumerate(files):
    temp_df = pd.read_csv(file)

    # Assign different year to avoid timestamp overlap
    year = 2023 + i
    temp_df["Date/Time"] = f"{year} " + temp_df["Date/Time"]

    # Fix 24:00:00 issue
    mask_24 = temp_df["Date/Time"].str.contains("24:00:00")

    temp_df.loc[mask_24, "Date/Time"] = (
        temp_df.loc[mask_24, "Date/Time"]
        .str.replace("24:00:00", "00:00:00", regex=False)
    )

    temp_df["Date/Time"] = pd.to_datetime(
        temp_df["Date/Time"], format="%Y %m/%d  %H:%M:%S"
    )

    temp_df.loc[mask_24, "Date/Time"] += pd.Timedelta(days=1)

    dfs.append(temp_df)

# Combine datasets
df = pd.concat(dfs)
df = df.sort_values("Date/Time").set_index("Date/Time")

print("Consolidated shape:", df.shape)

#target Feature from dataset
target_col = "Electricity:Facility [kW](Hourly)"
series = df[target_col]

print("Null values:", series.isnull().sum())

#eda plots
plt.figure(figsize=(12,4))
plt.plot(series)
plt.title("Full Electricity Consumption Time Series")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_full_series.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(series.iloc[:24*7])
plt.title("First Week Consumption Pattern")
plt.xlabel("Time (Hours)")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_first_week.png")
plt.close()

plt.figure(figsize=(10,4))
plt.plot(series.iloc[:48])
plt.title("First 48 Hours Consumption Pattern")
plt.xlabel("Time (Hours)")
plt.ylabel("Electricity Consumption (kW)")
plt.savefig("eda_48_hours.png")
plt.close()

plt.figure(figsize=(6,4))
plt.hist(series, bins=50)
plt.title("Electricity Consumption Distribution")
plt.xlabel("Electricity Consumption (kW)")
plt.ylabel("Frequency")
plt.savefig("eda_distribution.png")
plt.close()

plt.figure(figsize=(10,4))

plt.hist(series, bins=50)

plt.title("Distribution of Electricity Consumption")
plt.xlabel("Electricity Consumption (kW)")
plt.ylabel("Frequency")

plt.savefig("hist.png")
plt.close()

#Train test split
split_ratio = 0.8
split_index = int(len(series) * split_ratio)

train_series = series.iloc[:split_index]
test_series  = series.iloc[split_index:]

# scaling
scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_series.values.reshape(-1, 1))
test_scaled  = scaler.transform(test_series.values.reshape(-1, 1))

#Sequence creation
TIME_STEPS = 168   # Weekly seasonality

def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i:i + time_steps])
        y.append(data[i + time_steps])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, TIME_STEPS)
X_test, y_test   = create_sequences(test_scaled, TIME_STEPS)

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

#training or loading model
if TRAIN_MODEL or not os.path.exists("electricity_lstm_model.h5"):

    print("\nTraining Model...")

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, 1)),
        LSTM(32),
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
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=1
    )

    model.save("electricity_lstm_model.h5")

else:
    print("\nLoading Saved Model...")
    model = load_model("electricity_lstm_model.h5", compile=False)
    model.compile(optimizer="adam", loss="mse")
    history = None

#Prediction
y_pred = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test)
y_pred_inv = scaler.inverse_transform(y_pred)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# MSE
mse = mean_squared_error(y_test_inv, y_pred_inv)

# RMSE
rmse = np.sqrt(mse)

# MAE
mae = mean_absolute_error(y_test_inv, y_pred_inv)

# R² Score
r2 = r2_score(y_test_inv, y_pred_inv)

print("\n📊 LSTM Evaluation Metrics:")
print(f"MSE : {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")

# np.save("y_test.npy", y_test_inv)
# np.save("y_pred.npy", y_pred_inv)

np.save("lstm_pred.npy", y_pred_inv)
np.save("lstm_test.npy", y_test_inv)

# Evaluation
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
mae  = mean_absolute_error(y_test_inv, y_pred_inv)

#mape Calculation
epsilon = 1e-10  # small value to prevent division by zero
mape = np.mean(
    np.abs((y_test_inv - y_pred_inv) / (y_test_inv + epsilon))
) 

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("MAE:", mae)
print("MAPE:", mape, "%")

# Plots
plt.figure(figsize=(10,5))
plt.plot(y_test_inv, label="Actual")
plt.plot(y_pred_inv, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Electricity")
plt.savefig("actual_vs_predicted.png")
plt.close()

if history is not None:
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title("Training vs Validation Loss")
    plt.savefig("training_loss.png")
    plt.close()

print("\n✅ Model training, evaluation, and plots completed successfully.")

# pick random starting index
start_idx = np.random.randint(0, len(series) - TIME_STEPS - 1)

# slice 168 hours
random_168 = series.values[start_idx : start_idx + TIME_STEPS]

# actual next value
actual_next = series.values[start_idx + TIME_STEPS]

# scale
random_scaled = scaler.transform(random_168.reshape(-1, 1))
random_scaled = random_scaled.reshape(1, TIME_STEPS, 1)

# predict
pred_scaled = model.predict(random_scaled)
pred = scaler.inverse_transform(pred_scaled)

print("\n📊 RANDOM WINDOW PREDICTION")
print(f"Start Index: {start_idx}")
print(f"Predicted: {pred[0][0]:.2f} kW")
print(f"Actual   : {actual_next:.2f} kW")

#24 hour forecast(sequence of 168 hours to predict next 24 hours)
future_steps = 24

# pick random starting point
start_idx = np.random.randint(0, len(series) - TIME_STEPS - future_steps)

# get 168-hour input
input_seq = series.values[start_idx : start_idx + TIME_STEPS]

# actual future values (ground truth)
actual_future = series.values[start_idx + TIME_STEPS : start_idx + TIME_STEPS + future_steps]

# scale input
input_scaled = scaler.transform(input_seq.reshape(-1, 1))

predictions = []

for _ in range(future_steps):
    # reshape
    input_reshaped = input_scaled.reshape(1, TIME_STEPS, 1)

    # predict next step
    pred_scaled = model.predict(input_reshaped, verbose=0)

    # store
    predictions.append(pred_scaled[0][0])

    # update window
    input_scaled = np.append(input_scaled[1:], pred_scaled, axis=0)

# inverse transform
predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

print("\n RANDOM WINDOW 24-HOUR FORECAST")

for i in range(future_steps):
    print(f"Hour {i+1}: Predicted = {predictions[i][0]:.2f} kW | Actual = {actual_future[i]:.2f} kW")

    plt.figure(figsize=(10,4))

plt.plot(actual_future, label="Actual", marker='o')
plt.plot(predictions, label="Predicted", marker='x')

plt.title("Random Window: Next 24 Hours Forecast")
plt.xlabel("Hour Ahead")
plt.ylabel("Electricity Consumption (kW)")
plt.legend()

plt.savefig("random_24hr_forecast.png")
plt.close()