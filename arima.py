# ================= PURE ARIMA MODEL =================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ---------------- LOAD DATA ----------------
files = [
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_LOW.csv",
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_BASE.csv",
    "USA_GA_Albany-Dougherty.County.AP.722160_TMY3_HIGH.csv"
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

df = pd.concat(dfs)
df = df.sort_values("Date/Time").set_index("Date/Time")

# ---------------- TARGET ----------------
target_col = "Electricity:Facility [kW](Hourly)"
series = df[target_col]

# ---------------- TRAIN TEST SPLIT ----------------
split = int(len(series) * 0.8)
train = series[:split]
test = series[split:]

# ---------------- ARIMA MODEL ----------------
print("\nTraining ARIMA...")

model = ARIMA(train, order=(5,1,0))  # simple baseline
model_fit = model.fit()

# ---------------- FORECAST ----------------
forecast = model_fit.forecast(steps=len(test))

# ---------------- METRICS ----------------
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast)

print("\n📊 ARIMA RESULTS")
print("RMSE:", rmse)
print("MAE :", mae)

# ---------------- SAVE ----------------
np.save("arima_pred.npy", forecast)
np.save("arima_test.npy", test.values)

# ---------------- PLOT ----------------
plt.figure(figsize=(10,5))
plt.plot(test.values, label="Actual")
plt.plot(forecast, label="ARIMA Predicted")
plt.legend()
plt.title("ARIMA Forecast")
plt.savefig("Results_graphs/arima_result.png")
plt.close()

print("\n✅ ARIMA MODEL DONE")