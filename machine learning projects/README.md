# LSTM Stock Price Predictor

This project predicts the next-day closing price of a financial asset (e.g., AAPL) using an LSTM deep learning model.

## Features
- Downloads 5 years of historical stock data using yfinance
- Focuses on 'Close' price only
- Data normalization with MinMaxScaler
- 80/20 train/test split
- 60-day time-series sequences for prediction
- LSTM model built with TensorFlow/Keras
- Plots training/validation loss and actual vs predicted prices
- Saves trained model as `.h5`
- `main.py` for loading the model and predicting the next day's close

## Project Structure
```
├── src/           # Data loading, preprocessing, and model code
├── notebooks/     # EDA and prototyping
├── models/        # Saved models (.h5)
├── outputs/       # Plots and results
├── requirements.txt
├── README.md
└── main.py
```

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run training: `python src/train_lstm.py`
3. Predict next day: `python main.py`

See `notebooks/` for EDA and prototyping examples.
