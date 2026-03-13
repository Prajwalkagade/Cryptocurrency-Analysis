from prophet import Prophet
import pandas as pd

def train_prophet(df):
    dfp = pd.DataFrame({
        "ds": df.index,
        "y": df["close"].values
    })
    model = Prophet()
    model.fit(dfp)
    return model

def forecast_prophet(model, days=30):
    future = model.make_future_dataframe(periods=days)
    forecast = model.predict(future)
    return forecast
