from fastapi import FastAPI
from pydantic import BaseModel, Field
import numpy as np
import os
import joblib
from tensorflow.keras.models import load_model
from fastapi import HTTPException
import time
import logging
import zipfile
from fastapi.responses import FileResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Inicialização da API
# ----------------------------
app = FastAPI(
    title="API LSTM PETR4",
    description="API para prever o preço de fechamento de PETR4 usando modelo LSTM",
    version="1.0.0"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

modelo_path = os.path.join(BASE_DIR, "modelo_lstm_petr4.h5")
scaler_path = os.path.join(BASE_DIR, "scaler_petr4.save")

# Carregar modelo e scaler
modelo = load_model(modelo_path)
scaler = joblib.load(scaler_path)

TIME_STEP = 60

# ----------------------------
# Modelo de dados
# ----------------------------


class DadosEntrada(BaseModel):
    valores: list = Field(
        ...,
        description="Lista com os últimos 60 preços de fechamento",
        example=[
            81.1, 80.9, 81.0, 81.2, 81.3, 81.5, 81.4, 81.6, 81.7, 81.8,
            82.0, 81.9, 82.1, 82.2, 82.0, 81.8, 81.7, 81.9, 82.0, 82.1,
            82.2, 82.3, 82.4, 82.5, 82.6, 82.7, 82.8, 82.9, 83.0, 83.1,
            83.2, 83.3, 83.4, 83.5, 83.6, 83.7, 83.8, 83.9, 84.0, 84.1,
            84.2, 84.3, 84.4, 84.5, 84.6, 84.7, 84.8, 84.9, 85.0, 85.1,
            85.2, 85.3, 85.4, 85.5, 85.6, 85.7, 85.8, 85.9, 86.0, 86.1
        ]
    )

# ----------------------------
# Middleware de tempo de resposta
# ----------------------------


@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time

    logger.info(
        f"Endpoint: {request.url.path} | Tempo de resposta: {process_time:.4f}s"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response

# ----------------------------
# Endpoints
# ----------------------------


@app.get("/", summary="Verificar status da API", description="Retorna mensagem de que a API está ativa")
def home():
    return {"mensagem": "API LSTM PETR4 ativa"}


@app.post(
    "/prever",
    summary="Prever preço de fechamento",
    description="Recebe uma lista com os últimos 60 preços e retorna a previsão do próximo fechamento"
)
def prever(dados: DadosEntrada):

    if len(dados.valores) != TIME_STEP:
        raise HTTPException(
            status_code=400,
            detail=f"Envie exatamente {TIME_STEP} valores."
        )

    # Preparar dados para o modelo
    entrada = np.array(dados.valores).reshape(-1, 1)
    entrada_scaled = scaler.transform(entrada)
    entrada_scaled = entrada_scaled.reshape(1, TIME_STEP, 1)

    # Previsão
    previsao_scaled = modelo.predict(entrada_scaled)
    previsao_real = scaler.inverse_transform(previsao_scaled)

    return {
        "previsao_fechamento": float(previsao_real[0][0])
    }


OUTPUT_DIR = "previsoes"
os.makedirs(OUTPUT_DIR, exist_ok=True)
ZIP_NAME = "todas_previsoes.zip"


@app.get("/baixar",
         summary="Baixar todos os relatórios CSV",
         description="Gera dinamicamente um arquivo ZIP contendo todos os arquivos CSV presentes na pasta de previsões.",
         response_description="Arquivo ZIP contendo todos os relatórios gerados"
         )
def baixar_todos():

    arquivos_csv = [
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".csv")
    ]

    if not arquivos_csv:
        raise HTTPException(status_code=404, detail="Nenhum CSV encontrado")

    zip_path = os.path.join(OUTPUT_DIR, ZIP_NAME)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for arquivo in arquivos_csv:
            caminho_completo = os.path.join(OUTPUT_DIR, arquivo)
            zipf.write(caminho_completo, arcname=arquivo)

    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=ZIP_NAME
    )
