# 📈 Stock Price Predictor

A machine learning web app that predicts stock prices using trend analysis, EMA, and volatility modeling — with live Yahoo Finance data.

## 🚀 Live Demo
Deployed on Streamlit Cloud.

🔗 **Try the live application here:**  
https://stock-price-predictor-gs9papbjzqqegyhuhgr2ey.streamlit.app/

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
