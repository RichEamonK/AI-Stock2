pip install Flask

# app.py
from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm

# Import all functions from the buy score model
from unified_buy_score_model import *  # Assuming the model code is in a file named unified_buy_score_model.py

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/data')
def get_data():
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "XOM", "NVDA"]
    results = unified_buy_score(tickers)
    return jsonify(results.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
