import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import os

# Criar pasta models se não existir
if not os.path.exists("models"):
    os.makedirs("models")

# Baixar dados PETR4
df = yf.download("PETR4.SA", start="2015-01-01")

data = df[['Close']].values

# Normalizar
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# Criar janelas
X = []
y = []

window_size = 60

for i in range(window_size, len(data_scaled)):
    X.append(data_scaled[i-window_size:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)

# Reshape para LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Modelo
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Treino
model.fit(X, y, epochs=10, batch_size=32)

# Salvar modelo
model.save("models/petr4_lstm.keras")

print("Modelo salvo em models/petr4_lstm.keras")
