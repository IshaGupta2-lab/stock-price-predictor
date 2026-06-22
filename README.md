# 📈 Stock Price Predictor

A machine learning web app that predicts stock prices using trend analysis, EMA, and volatility modeling — with live Yahoo Finance data.

## 🚀 Live Demo
Deployed on Streamlit Cloud.

## ✨ Features
- 📊 Live stock data via Yahoo Finance (`yfinance`)
- 📈 Price prediction with confidence band
- 📉 Technical indicators — SMA, EMA, RSI, MACD
- 📋 Day-by-day forecast table with BUY/SELL signals
- 🌐 Works for US stocks (AAPL, TSLA) and Indian stocks (RELIANCE.NS, TCS.NS)

## 📦 Setup Locally

```bash
git clone <your-repo-url>
cd stock-app
pip install -r requirements.txt
streamlit run app.py
```

## 🌐 Deploy on Streamlit Cloud
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect GitHub repo → set main file as `app.py`
4. Click Deploy!

## ⚠️ Disclaimer
This app is for educational purposes only. Never make real investment decisions based solely on ML predictions.