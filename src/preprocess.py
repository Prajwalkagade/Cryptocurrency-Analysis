import pandas as pd

def resample_fill(df):
    df = df.resample("D").last().ffill()
    return df

def add_features(df):
    df["returns"] = df["close"].pct_change()
    df["volatility_30"] = df["returns"].rolling(30).std()
    df["rolling_mean_7"] = df["close"].rolling(7).mean()
    return df


