import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def train_lstm(df, look=60, epochs=5):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["close"]])

    X, y = [], []
    for i in range(look, len(scaled)):
        X.append(scaled[i-look:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(look, 1)),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)

    return model, scaler
