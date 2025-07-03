from flask import Flask, request, render_template
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import io
import base64
from sklearn.ensemble import RandomForestClassifier
from ta.momentum import RSIIndicator
from ta.trend import MACD

app = Flask(__name__)

def generate_features(data):
    data['RSI'] = RSIIndicator(data['Close']).rsi()
    macd = MACD(data['Close'])
    data['MACD'] = macd.macd_diff()
    data = data.dropna()
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    return data

def train_model(data):
    features = ['RSI', 'MACD']
    X = data[features]
    y = data['Target']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def plot_data(data):
    fig, ax = plt.subplots()
    data['Close'].plot(ax=ax)
    ax.set_title("Stock Closing Price")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.route("/", methods=["GET", "POST"])
def index():
    plot_url = ""
    prediction = ""
    if request.method == "POST":
        symbol = request.form["symbol"]
        data = yf.download(symbol, period="6mo", interval="1d")
        data = generate_features(data)
        model = train_model(data)
        latest_data = data[['RSI', 'MACD']].iloc[-1:]
        prediction = "UP" if model.predict(latest_data)[0] == 1 else "DOWN"
        plot_url = plot_data(data)

    return render_template("index.html", plot_url=plot_url, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)