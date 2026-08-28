import os

import joblib
import numpy as np
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TICKER = "ITUB4.SA"
START_DATE = "2015-01-01"

WINDOW_SIZE = 60
TRAIN_SIZE = 0.80

EPOCHS = 10
BATCH_SIZE = 32

MODEL_DIR = "models"

MODEL_PATH = os.path.join(MODEL_DIR, "itub4_lstm.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_itub4.save")


# ============================================================
# PREPARAÇÃO
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)

print("=" * 60)
print("TREINAMENTO LSTM - ITUB4")
print("=" * 60)

print(f"\nBaixando dados de {TICKER}...")

df = yf.download(
    TICKER,
    start=START_DATE,
    auto_adjust=False
)

if df.empty:
    raise RuntimeError("Não foi possível baixar os dados da ação.")

data = df[["Close"]].values

print(f"Quantidade de registros: {len(data)}")


# ============================================================
# NORMALIZAÇÃO
# ============================================================

scaler = MinMaxScaler(feature_range=(0, 1))

data_scaled = scaler.fit_transform(data)


# ============================================================
# CRIAÇÃO DAS JANELAS
# ============================================================

X = []
y = []

for i in range(WINDOW_SIZE, len(data_scaled)):
    X.append(data_scaled[i - WINDOW_SIZE:i, 0])
    y.append(data_scaled[i, 0])

X = np.array(X)
y = np.array(y)

X = np.reshape(
    X,
    (X.shape[0], X.shape[1], 1)
)

print(f"\nTotal de amostras: {len(X)}")


# ============================================================
# DIVISÃO TREINO / TESTE
# ============================================================

split_index = int(len(X) * TRAIN_SIZE)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

print(f"Amostras de treino: {len(X_train)}")
print(f"Amostras de teste:  {len(X_test)}")


# ============================================================
# MODELO LSTM
# ============================================================

print("\nCriando modelo LSTM...")

model = Sequential()

model.add(
    LSTM(
        50,
        return_sequences=False,
        input_shape=(X_train.shape[1], 1)
    )
)

model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)


# ============================================================
# TREINAMENTO
# ============================================================

print("\nIniciando treinamento...")

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)


# ============================================================
# PREVISÃO NO CONJUNTO DE TESTE
# ============================================================

print("\nRealizando previsões no conjunto de teste...")

predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# CONVERSÃO PARA PREÇO REAL
# ============================================================

y_test_real = scaler.inverse_transform(
    y_test.reshape(-1, 1)
)

predictions_real = scaler.inverse_transform(
    predictions_scaled
)


# ============================================================
# MÉTRICAS
# ============================================================

mae = mean_absolute_error(
    y_test_real,
    predictions_real
)

rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        predictions_real
    )
)

mape = np.mean(
    np.abs(
        (y_test_real - predictions_real)
        / y_test_real
    )
) * 100


# ============================================================
# RESULTADOS
# ============================================================

print("\n" + "=" * 60)
print("RESULTADOS DO MODELO")
print("=" * 60)

print(f"MAE : R$ {mae:.4f}")
print(f"RMSE: R$ {rmse:.4f}")
print(f"MAPE: {mape:.2f}%")

print("=" * 60)


# ============================================================
# SERIALIZAÇÃO
# ============================================================

print("\nSalvando modelo...")

model.save(MODEL_PATH)

print(f"Modelo salvo em: {MODEL_PATH}")

print("\nSalvando scaler...")

joblib.dump(
    scaler,
    SCALER_PATH
)

print(f"Scaler salvo em: {SCALER_PATH}")

print("\nTreinamento concluído com sucesso!")
