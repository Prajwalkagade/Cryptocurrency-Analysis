from pycoingecko import CoinGeckoAPI
import pandas as pd

cg = CoinGeckoAPI()

def fetch_historical_ohlc(coin="bitcoin", vs="usd", days=30):
    """
    Fetch OHLC + volume using hourly data (24h × days)
    Generates:
        open, high, low, close, volume
    """
    data = cg.get_coin_market_chart_by_id(
        id=coin, vs_currency=vs, days=days
    )

    df = pd.DataFrame({
        "ts": pd.to_datetime([x[0] for x in data["prices"]], unit="ms"),
        "price": [x[1] for x in data["prices"]],
        "volume": [v[1] for v in data["total_volumes"]]
    })

    # Convert hourly → 1-day OHLC
    df = df.set_index("ts").resample("D").agg({
        "price": ["first", "max", "min", "last"],
        "volume": "sum"
    })

    df.columns = ["open", "high", "low", "close", "volume"]
    return df