from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).parent / "data" / "Bitstamp_BTCUSD_d.csv"

FEATURE_COLUMNS = [
    "daily_return",
    "daily_range",
    "close_vs_ma_7",
    "volatility_7",
]


def load_data(path=DATA_PATH):
    """Load the bitcoin price data, oldest day first."""
    # The first line of the file is a website link, so skip it.
    df = pd.read_csv(path, skiprows=1)
    df = df.rename(columns={"Volume BTC": "volume"})
    df["datetime_utc"] = pd.to_datetime(df["unix"], unit="s", utc=True)

    # The newest row is the day the file was downloaded.
    # That day wasn't over yet, so remove it.
    df = df.sort_values("datetime_utc").reset_index(drop=True)
    df = df.iloc[:-1].copy()
    return df[["datetime_utc", "open", "high", "low", "close", "volume"]]


def add_features(df):
    """Add the answer column (target) and the columns the model learns from."""
    df = df.copy()
    traded = df["volume"] > 0

    # Target: 1 if the next day closed higher, 0 if not.
    # Skip it if the next day is missing or there was no trading.
    is_tomorrow = df["datetime_utc"].shift(-1) - df["datetime_utc"] == pd.Timedelta(days=1)
    valid_target = is_tomorrow & traded & traded.shift(-1, fill_value=False)
    df["next_close"] = df["close"].shift(-1).where(valid_target)
    df["target"] = (
        (df["next_close"] > df["close"])
        .astype("Int64")
        .where(df["next_close"].notna())
    )

    # Daily return: how much the price changed since yesterday.
    consecutive_days = df["datetime_utc"].diff() == pd.Timedelta(days=1)
    both_days_traded = traded & traded.shift(1, fill_value=False)
    df["daily_return"] = (
        df["close"].pct_change(fill_method=None)
        .where(consecutive_days & both_days_traded)
    )

    df["daily_range"] = ((df["high"] - df["low"]) / df["close"]).where(traded)

    # 7-day average: needs 7 days in a row with trading.
    seven_calendar_days = (
        df["datetime_utc"] - df["datetime_utc"].shift(6)
    ) == pd.Timedelta(days=6)
    df["ma_7"] = (
        df["close"].where(traded)
        .rolling(window=7, min_periods=7).mean()
        .where(seven_calendar_days)
    )
    df["close_vs_ma_7"] = df["close"] / df["ma_7"] - 1

    df["volatility_7"] = df["daily_return"].rolling(window=7, min_periods=7).std()
    return df
