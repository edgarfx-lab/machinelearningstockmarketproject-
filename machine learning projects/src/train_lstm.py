import os
import numpy as np
import matplotlib.pyplot as plt
from data_utils import download_data, normalize_data, create_sequences
from model_utils import build_lstm, plot_loss
from sklearn.model_selection import train_test_split

TICKER = 'AAPL'
SEQ_LEN = 60
MODEL_PATH = '../models/lstm_model.h5'
PLOT_PATH = '../outputs/loss_plot.png'

# Download and preprocess data
df = download_data(TICKER)
scaled, scaler = normalize_data(df)
X, y = create_sequences(scaled, SEQ_LEN)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Build and train model
model = build_lstm((SEQ_LEN, 1))
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model and plot
os.makedirs('../models', exist_ok=True)
os.makedirs('../outputs', exist_ok=True)
model.save(MODEL_PATH)
plot_loss(history, PLOT_PATH)

# Predict and plot
preds = model.predict(X_test)
preds_rescaled = scaler.inverse_transform(preds)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

plt.figure(figsize=(10,5))
plt.plot(y_test_rescaled, label='Actual')
plt.plot(preds_rescaled, label='Predicted')
plt.legend()
plt.title('Actual vs Predicted Close Price')
plt.savefig('../outputs/actual_vs_predicted.png')
plt.close()
