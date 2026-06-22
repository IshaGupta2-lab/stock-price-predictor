import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.up-box   { background:#00c85322; border:2px solid #00c853; border-radius:10px; padding:20px; text-align:center; }
.down-box { background:#ff4b4b22; border:2px solid #ff4b4b; border-radius:10px; padding:20px; text-align:center; }
.big-text { font-size:2rem; font-weight:bold; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Predictor")
st.markdown("**Trend-based prediction model for stock price forecasting**")
st.markdown("---")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    ticker  = st.text_input("Stock Ticker Symbol", value="AAPL").upper()
    period  = st.selectbox("Training Data Period", ["1y","2y","5y"], index=1)
    n_days  = st.slider("Days to Predict", 7, 60, 30)
    st.markdown("---")
    st.markdown("**Popular Tickers:**")
    st.markdown("AAPL · GOOGL · MSFT · TSLA · AMZN · RELIANCE.NS · TCS.NS · INFY.NS")
    st.markdown("---")
    run_btn = st.button("Run Prediction", use_container_width=True, type="primary")
    st.info("Data is fetched live from Yahoo Finance")

tab1, tab2, tab3 = st.tabs(["Prediction", "Technical Analysis", "Model Info"])

# ── Generate synthetic stock data ─────────────────────────────────────────────
def generate_stock_data(ticker, period):
    np.random.seed(abs(hash(ticker)) % (2**31))
    days_map = {"1y": 252, "2y": 504, "5y": 1260}
    n  = days_map.get(period, 504)
    price_map = {
        "AAPL": 150, "GOOGL": 140, "MSFT": 380, "TSLA": 200,
        "AMZN": 180, "META": 500, "NVDA": 800, "NFLX": 600,
        "RELIANCE.NS": 2800, "TCS.NS": 3900, "INFY.NS": 1500,
    }
    S0      = price_map.get(ticker, 100)
    mu      = 0.0003
    sigma   = 0.018
    returns = np.random.normal(mu, sigma, n)
    prices  = S0 * np.exp(np.cumsum(returns))
    dates   = pd.date_range(end=pd.Timestamp.today(), periods=n, freq="B")
    df      = pd.DataFrame({"Close": prices,
                             "Open":  prices * (1 + np.random.uniform(-0.01, 0.01, n)),
                             "High":  prices * (1 + np.abs(np.random.uniform(0, 0.02, n))),
                             "Low":   prices * (1 - np.abs(np.random.uniform(0, 0.02, n))),
                             "Volume": np.random.randint(10_000_000, 100_000_000, n)},
                            index=dates)
    return df

# ── Prediction function (fixed) ───────────────────────────────────────────────
def predict_prices(close_series, n_days):
    prices = np.array(close_series.values, dtype=float).flatten()

    # Need at least 61 data points
    if len(prices) < 61:
        prices = np.pad(prices, (61 - len(prices), 0), mode='edge')

    # Trend from last 30 days
    last_30 = prices[-30:]
    x       = np.arange(len(last_30))
    slope   = np.polyfit(x, last_30, 1)[0]
    trend   = slope / (last_30[-1] + 1e-9)

    # Volatility from last 60 days
    hist_60  = prices[-61:]                         # 61 points → 60 returns
    rets_60  = np.diff(hist_60) / (hist_60[:-1] + 1e-9)
    vol      = float(np.std(rets_60))

    # EMA
    ema = pd.Series(prices).ewm(span=20).mean().values

    # Generate future prices
    preds   = []
    last_p  = float(prices[-1])
    np.random.seed(42)
    for _ in range(n_days):
        noise  = float(np.random.normal(trend, vol))
        last_p = last_p * (1.0 + noise)
        preds.append(last_p)

    last_date    = close_series.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                  periods=n_days, freq="B")
    return np.array(preds), future_dates, ema, vol

# ── Fetch data ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Fetching stock data...")
def get_data(ticker, period):
    try:
        import yfinance as yf
        df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
        if df.empty:
            raise ValueError("empty")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df, "live"
    except Exception:
        return generate_stock_data(ticker, period), "synthetic"

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    with st.spinner(f"Loading {ticker} data..."):
        df, data_source = get_data(ticker, period)

    if data_source == "synthetic":
        st.warning(f"Live data unavailable — showing simulated data for **{ticker}**.")
    else:
        st.success(f"Live data loaded for **{ticker}**")

    # Extract close prices safely
    if "Close" in df.columns:
        close = df["Close"].squeeze()
    else:
        close = df.iloc[:, 0].squeeze()

    close = pd.Series(close.values.flatten(), index=df.index, name="Close")

    # Run prediction
    preds, future_dates, ema, vol = predict_prices(close, n_days)

    current_price  = float(close.iloc[-1])
    predicted_last = float(preds[-1])
    change_pct     = (predicted_last - current_price) / current_price * 100

    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price",          f"${current_price:.2f}")
    c2.metric(f"Predicted ({n_days}d)", f"${predicted_last:.2f}", f"{change_pct:+.2f}%")
    c3.metric("52W High",               f"${float(close.max()):.2f}")
    c4.metric("52W Low",                f"${float(close.min()):.2f}")

    st.markdown("---")

    # Direction box
    if predicted_last > current_price:
        st.markdown(f"""<div class="up-box">
            <div class="big-text">BULLISH — Price Expected to Rise</div>
            <p style="font-size:1.1rem">Predicted change: <b>+{change_pct:.2f}%</b> over {n_days} days</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="down-box">
            <div class="big-text">BEARISH — Price Expected to Fall</div>
            <p style="font-size:1.1rem">Predicted change: <b>{change_pct:.2f}%</b> over {n_days} days</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"{ticker} — Historical + {n_days}-Day Forecast")

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")

    hist = close.iloc[-120:]
    ax.plot(hist.index, hist.values, color="#4C9BE8", lw=2, label="Historical Price")

    ema_s = pd.Series(ema[-120:], index=close.index[-120:])
    ax.plot(ema_s.index, ema_s.values, color="#FFA500", lw=1.5,
            linestyle="--", label="EMA (20)")

    ax.plot(future_dates, preds, color="#00c853", lw=2.5,
            linestyle="--", label=f"{n_days}-Day Forecast")

    scale  = np.sqrt(np.arange(1, n_days+1) / n_days)
    upper  = preds * (1 + 1.5 * vol * scale)
    lower  = preds * (1 - 1.5 * vol * scale)
    ax.fill_between(future_dates, lower, upper, alpha=0.15,
                    color="#00c853", label="Confidence Band")

    ax.axvline(close.index[-1], color="gray", linestyle=":", lw=1.5)
    ax.set_title(f"{ticker} Stock Price Prediction", color="white", fontsize=14)
    ax.set_xlabel("Date", color="white")
    ax.set_ylabel("Price", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1e1e2e", labelcolor="white")
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    plt.xticks(rotation=30)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.subheader("Day-by-Day Forecast")
    forecast_df = pd.DataFrame({
        "Date":              [d.strftime("%Y-%m-%d") for d in future_dates],
        "Predicted Price":   [f"${p:.2f}" for p in preds],
        "Change from Today": [f"{(p-current_price)/current_price*100:+.2f}%" for p in preds],
        "Signal":            ["BUY" if p > current_price else "SELL" for p in preds],
    })
    st.dataframe(forecast_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TECHNICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Technical Indicators")

    df2, _ = get_data(ticker, period)
    if "Close" in df2.columns:
        close2 = df2["Close"].squeeze()
    else:
        close2 = df2.iloc[:, 0].squeeze()
    close2 = pd.Series(close2.values.flatten(), index=df2.index, name="Close")
    price2 = close2.iloc[-120:]

    sma20 = price2.rolling(20).mean()
    sma50 = price2.rolling(50).mean()

    delta = price2.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rsi   = 100 - (100 / (1 + gain / (loss + 1e-9)))

    ema12  = price2.ewm(span=12).mean()
    ema26  = price2.ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    rsi_val  = float(rsi.iloc[-1])
    macd_val = float(macd.iloc[-1])
    c1, c2, c3 = st.columns(3)
    c1.metric("RSI (14)", f"{rsi_val:.1f}",
              "Overbought" if rsi_val > 70 else ("Oversold" if rsi_val < 30 else "Neutral"))
    c2.metric("MACD", f"{macd_val:.3f}", "Bullish" if macd_val > 0 else "Bearish")
    c3.metric("SMA20 vs SMA50",
              "Bullish" if float(sma20.iloc[-1]) > float(sma50.iloc[-1]) else "Bearish")

    st.markdown("---")

    fig2, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    fig2.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#0e1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    axes[0].plot(price2.index, price2.values, color="#4C9BE8", lw=2, label="Price")
    axes[0].plot(sma20.index,  sma20.values,  color="#FFA500", lw=1.5, linestyle="--", label="SMA 20")
    axes[0].plot(sma50.index,  sma50.values,  color="#e74c3c", lw=1.5, linestyle="--", label="SMA 50")
    axes[0].set_title(f"{ticker} Price & Moving Averages", color="white")
    axes[0].set_ylabel("Price", color="white")
    axes[0].legend(facecolor="#1e1e2e", labelcolor="white")

    axes[1].plot(rsi.index, rsi.values, color="#9b59b6", lw=2)
    axes[1].axhline(70, color="#e74c3c", linestyle="--", lw=1)
    axes[1].axhline(30, color="#00c853", linestyle="--", lw=1)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("RSI (14)", color="white")
    axes[1].set_ylabel("RSI", color="white")

    axes[2].plot(macd.index,   macd.values,   color="#4C9BE8", lw=2,   label="MACD")
    axes[2].plot(signal.index, signal.values, color="#FFA500", lw=1.5, label="Signal")
    hist_vals = (macd - signal).values
    axes[2].bar(macd.index, hist_vals,
                color=["#00c853" if v >= 0 else "#e74c3c" for v in hist_vals], alpha=0.5)
    axes[2].axhline(0, color="gray", lw=0.8)
    axes[2].set_title("MACD", color="white")
    axes[2].set_ylabel("MACD", color="white")
    axes[2].legend(facecolor="#1e1e2e", labelcolor="white")
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    plt.xticks(rotation=30, color="white")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How the Prediction Works")
    st.markdown("""
    | Component | Description |
    |---|---|
    | **EMA (20)** | Exponential Moving Average — captures underlying trend |
    | **Momentum** | Linear trend slope from last 30 days |
    | **Volatility** | Std deviation of last 60 days returns |
    | **Confidence Band** | ±1.5σ band, widens further into future |

    ### Technical Indicators
    | Indicator | Signal |
    |---|---|
    | **RSI > 70** | Overbought — possible price drop |
    | **RSI < 30** | Oversold — possible price rise |
    | **MACD > 0** | Bullish momentum |
    | **SMA20 > SMA50** | Uptrend (Golden Cross) |

    ### Disclaimer
    > This app is for **educational purposes only**.
    > Never make real investment decisions based solely on ML predictions.
    """)
