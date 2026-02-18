from fastapi import FastAPI
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
import uvicorn

app = FastAPI()

# Carregar modelo
model = load_model("models/petr4_lstm.keras")


@app.get("/")
def home():
    return {"message": "API LSTM PETR4 rodando 🚀"}


@app.get("/predict")
def predict():

    # Baixar últimos dados
    df = yf.download("PETR4.SA", period="90d")
    data = df[['Close']].values

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # pegar últimos 60 dias
    last_60 = data_scaled[-60:]
    X_test = np.reshape(last_60, (1, 60, 1))

    prediction = model.predict(X_test)
    predicted_price = scaler.inverse_transform(prediction)

    return {
        "predicted_next_close": float(predicted_price[0][0])
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
