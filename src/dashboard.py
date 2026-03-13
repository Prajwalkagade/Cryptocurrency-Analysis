import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from data_fetch import fetch_historical_ohlc
from preprocess import add_features
from arima_model import train_arima, forecast_arima
from prophet_model import train_prophet, forecast_prophet
from lstm_model import train_lstm
from sentiment import sentiment

theme = st.sidebar.selectbox(
    "Theme",
    ["Dark", "Light"]
)
def apply_theme(theme):
    if theme == "Dark":
        st.markdown("""
            <style>
            .stApp {
                background-color: #0E1117;
                color: #FAFAFA;
            }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <style>
            .stApp {
                background-color: #FFFFFF;
                color: #000000;
            }
            </style>
        """, unsafe_allow_html=True)
apply_theme(theme)


st.set_page_config(layout="wide")
st.title(" Cryptocurrency Dashboard")

# SIDEBAR INPUTS

coin = st.sidebar.selectbox("Select Coin", ["bitcoin", "ethereum", "solana", "dogecoin"])
days = st.sidebar.slider("Historical Days", 30, 365, 180)
future_days = st.sidebar.slider("Forecast Future Days", 7, 90, 30)

# LOAD DATA

df = fetch_historical_ohlc(coin, "usd", days)
df = add_features(df)

# ARIMA MODEL

st.subheader("ARIMA")
try:
    arima_model = train_arima(df)
    arima_pred = forecast_arima(arima_model, steps=future_days)

    fig_arima = px.line(arima_pred, y="mean", title="ARIMA Model")
    st.plotly_chart(fig_arima, width="stretch")
except Exception as e:
    st.error(f"ARIMA Error: {e}")

# PROPHET MODEL

st.subheader("Prophet Forecast")
try:
    prophet_model = train_prophet(df)
    prophet_pred = forecast_prophet(prophet_model, future_days)

    fig_prophet = px.line(prophet_pred, x="ds", y="yhat", title="Prophet Forecast")
    st.plotly_chart(fig_prophet, width="stretch")
except Exception as e:
    st.error(f"Prophet Error: {e}")

# LSTM MODEL
st.subheader(" LSTM Model")
try:
    lstm_model, scaler = train_lstm(df)

    st.success("LSTM successfully trained!")

except Exception as e:
    st.error(f"LSTM Error: {e}")

#  SENTIMENT ANALYSIS
st.subheader("Sentiment Analysis (News / Tweets)")
sent_input = st.text_area("Enter crypto-related news or tweet:")

if st.button("Analyze Sentiment"):
    score = sentiment(sent_input)
    st.write("Sentiment Score:", score)
    if score > 0:
        st.success("Market Sentiment: Positive 😊")
    elif score < 0:
        st.error("Market Sentiment: Negative 😡")
    else:
        st.warning("Market Sentiment: Neutral 😐")

# 5️⃣ CANDLESTICK CHART
st.subheader("Candlestick Price Chart")
fig_candle = go.Figure(data=[
    go.Candlestick(
        x=df.index,
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"]
    )
])
st.plotly_chart(fig_candle, width="stretch")

# 6️⃣ VOLUME BAR CHART

st.subheader("Volume")
fig_vol = px.bar(df, x=df.index, y="volume", title="Trading Volume")
st.plotly_chart(fig_vol, width="stretch")

#  PIE CHART
st.subheader("Volume Distribution by Weekday (Pie Chart)")
df["day"] = df.index.strftime("%A")
fig_pie = px.pie(df, names="day", values="volume")
st.plotly_chart(fig_pie, width="stretch")

# DONUT CHART
st.subheader("Gain vs Loss Days (Donut Chart)")
df["return_type"] = ["Gain" if r > 0 else "Loss" for r in df["returns"].fillna(0)]
fig_donut = px.pie(df, names="return_type", hole=0.5)
st.plotly_chart(fig_donut, width="stretch")

#  RETURNS CHART
st.subheader("Daily Return Rate")
fig_returns = px.line(df, y="returns")
st.plotly_chart(fig_returns, width="stretch")

# MOVING AVERAGES
df["MA7"] = df["close"].rolling(7).mean()
df["MA30"] = df["close"].rolling(30).mean()
st.subheader("Moving Averages (7 & 30)")
fig_ma = px.line(df, y=["close", "MA7", "MA30"])
st.plotly_chart(fig_ma, width="stretch")

# 8️⃣ VOLATILITY
st.subheader("Volatility (30-day Rolling)")
fig_volatility = px.line(df, y="volatility_30")
st.plotly_chart(fig_volatility, width="stretch")
# 1️⃣2️⃣ CORRELATION HEATMAP
st.subheader("Correlation Heatmap")
corr = df[["open", "high", "low", "close", "volume", "returns", "volatility_30"]].corr()
fig_heat = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
st.plotly_chart(fig_heat, width="stretch")