import pandas as pd
import matplotlib.pyplot as plt

from features import load_data

# Load the data (same steps the model uses).
df = load_data()

# Check the data for problems.
print("\nDataset shape:")
print(df.shape)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDates in chronological order:")
print(df["datetime_utc"].is_monotonic_increasing)

print("\nDuplicate dates:")
print(df["datetime_utc"].duplicated().sum())

print("\nTime between consecutive rows:")
print(df["datetime_utc"].diff().value_counts())

expected_dates = pd.date_range(
    df["datetime_utc"].min(), df["datetime_utc"].max(), freq="D"
)
print("\nMissing calendar dates:")
print(expected_dates.difference(df["datetime_utc"]))

print("\nNumeric summary:")
print(df.describe().to_string())

print("\nZero-volume days:")
print((df["volume"] == 0).sum())

print("\nDate range:")
print(df["datetime_utc"].min())
print(df["datetime_utc"].max())

# Daily return. Skip it if the day before is missing.
consecutive_days = df["datetime_utc"].diff() == pd.Timedelta(days=1)
df["daily_return"] = df["close"].pct_change(fill_method=None).where(consecutive_days)

# Chart 1: closing price over time
plt.figure(figsize=(12, 5))
plt.plot(df["datetime_utc"], df["close"])
plt.title("Bitcoin Daily Closing Price on Bitstamp")
plt.xlabel("Date")
plt.ylabel("Closing Price (USD)")
plt.grid(alpha=0.3)
plt.tight_layout()

# Chart 2: trading volume
plt.figure(figsize=(12, 5))
plt.plot(df["datetime_utc"], df["volume"], linewidth=0.7)

plt.title("Bitcoin Daily Trading Volume on Bitstamp")
plt.xlabel("Date")
plt.ylabel("Volume (BTC)")
plt.grid(alpha=0.3)

plt.tight_layout()

# Chart 3: daily returns
plt.figure(figsize=(12, 5))
plt.plot(df["datetime_utc"], df["daily_return"] * 100, linewidth=0.7)

plt.title("Bitcoin Daily Returns")
plt.xlabel("Date")
plt.ylabel("Daily Return (%)")
plt.axhline(0, color="black", linewidth=0.8)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.show()
