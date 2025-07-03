import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD

st.title("📈 AI Stock Trend Predictor")

symbol = st.text_input("Enter Stock Symbol (e.g., AAPL, MSFT):", "AAPL")

if symbol:
    data = yf.download(symbol, period="6mo", interval="1d")

    if not data.empty:
        st.subheader(f"Closing Price for {symbol}")
        st.line_chart(data['Close'].squeeze())


        # Generate technical indicators
        data['RSI'] = RSIIndicator(data['Close']).rsi()
        macd = MACD(data['Close'])
        data['MACD'] = macd.macd_diff()
        data = data.dropna()
        data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)

        # Train model
        X = data[['RSI', 'MACD']]
        y = data['Target']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict next day movement
        latest_data = X.iloc[-1:]
        prediction = model.predict(latest_data)[0]
        prediction_label = "📈 UP" if prediction == 1 else "📉 DOWN"

        st.subheader("Prediction for Next Day")
        st.success(f"The model predicts the stock will go: **{prediction_label}**")

        # Plot RSI and MACD
        st.subheader("RSI and MACD Indicators")
        fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        ax[0].plot(data.index, data['RSI'], label='RSI', color='purple')
        ax[0].axhline(70, color='red', linestyle='--')
        ax[0].axhline(30, color='green', linestyle='--')
        ax[0].set_title("RSI")
        ax[1].plot(data.index, data['MACD'], label='MACD', color='blue')
        ax[1].set_title("MACD")
        st.pyplot(fig)
    else:
        st.error("No data found. Check the stock symbol.")
