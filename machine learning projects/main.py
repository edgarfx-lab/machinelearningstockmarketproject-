import numpy as np
import yfinance as yf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

TICKER = 'AAPL'
SEQ_LEN = 60
MODEL_PATH = 'models/lstm_model.h5'

def fetch_latest_data(ticker, seq_len):
    df = yf.download(ticker, period=f'{seq_len+1}d')
    close = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(close)
    return scaled, scaler

def main():
    scaled, scaler = fetch_latest_data(TICKER, SEQ_LEN)
    X_input = scaled[-SEQ_LEN:]
    X_input = X_input.reshape((1, SEQ_LEN, 1))
    model = load_model(MODEL_PATH)
    pred_scaled = model.predict(X_input)
    pred = scaler.inverse_transform(pred_scaled)
    print(f"Predicted next-day closing price for {TICKER}: ${pred[0,0]:.2f}")

if __name__ == '__main__':
    main()
