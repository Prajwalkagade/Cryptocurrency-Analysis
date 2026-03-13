
from statsmodels.tsa.arima.model import ARIMA

def train_arima(df, order=(5, 1, 0)):
    model = ARIMA(df["close"], order=order)
    return model.fit()

def forecast_arima(model, steps=30):
    return model.get_forecast(steps=steps).summary_frame()
