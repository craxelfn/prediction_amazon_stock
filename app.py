from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import os

app = Flask(__name__)

model = load_model("best_model.h5")

scaler = MinMaxScaler(feature_range=(0, 1))

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        df = pd.read_csv(file)

        if df.empty:
            return jsonify({'error': 'Empty CSV file'}), 400

        if 'Close' not in df.columns:
            return jsonify({'error': 'Missing expected column "Close" in CSV file'}), 400

        df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

        df = df.dropna(subset=['Close'])

        if df.empty:
            return jsonify({'error': 'No valid numeric data found in the stock prices column'}), 400

        prices = df['Close'].values.reshape(-1, 1)

        prices_normalized = scaler.fit_transform(prices)

        X_input = prices_normalized[-60:].reshape(1, 60, 1)

        predictions = []
        for _ in range(10):
            predicted_price = model.predict(X_input)
            predictions.append(predicted_price[0][0])

            X_input = np.append(X_input[:, 1:, :], predicted_price.reshape(1, 1, 1), axis=1)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

        predicted_prices = predicted_prices.flatten().tolist()

        future_dates = pd.date_range(df['Date'].iloc[-1], periods=11, freq='D')[1:].strftime('%Y-%m-%d').tolist()

        return jsonify({
            'predicted_prices': predicted_prices,
            'dates': future_dates
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
