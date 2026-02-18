import yfinance as yf
import requests
import pandas as pd
import os
from datetime import datetime

# --- CONFIGURAÇÃO ---
SIMBOLOS = ["PETR4.SA", "VALE3.SA"]  # Ações que você quer prever
NUM_DIAS = 60
URL_API = "https://techchallenge04.onrender.com/prever"  # Endpoint da sua API
OUTPUT_DIR = "previsoes"                   # Pasta onde os CSVs serão salvos

# Cria a pasta de saída se não existir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- FUNÇÕES ---


def pegar_ultimos_fechamentos(simbolo):
    """
    Pega os últimos NUM_DIAS fechamentos de uma ação.
    Sempre garante que fechamentos é uma lista de floats.
    """
    df = yf.download(simbolo, period="100d", interval="1d")

    # Confirma se a coluna 'Close' existe
    if 'Close' not in df.columns:
        raise ValueError(
            f"{simbolo}: coluna 'Close' não encontrada no DataFrame.")

    # Pega só a coluna Close, garante Series e transforma em lista
    # <- aqui o .values transforma em numpy array antes do tolist
    fechamentos = df['Close'].dropna().values.tolist()

    ultimos_valores = fechamentos[-NUM_DIAS:]
    if len(ultimos_valores) != NUM_DIAS:
        raise ValueError(
            f"{simbolo}: não foi possível obter {NUM_DIAS} valores válidos. Pegamos {len(ultimos_valores)}")

    return ultimos_valores


def chamar_api(valores):
    """Chama o endpoint /prever da API"""
    dados = {"valores": valores}
    resposta = requests.post(URL_API, json=dados)
    return resposta.json()


# --- EXECUÇÃO ---
previsoes = []

for simbolo in SIMBOLOS:
    try:
        valores = pegar_ultimos_fechamentos(simbolo)
        resultado = chamar_api(valores)
        previsao_fechamento = resultado["previsao_fechamento"]
        print(
            f"Ação: {simbolo} | Previsão fechamento: {previsao_fechamento:.2f}")
        # Guarda no DataFrame
        previsoes.append({
            "acao": simbolo,
            "data": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previsao_fechamento": previsao_fechamento
        })
    except Exception as e:
        print(f"Erro para {simbolo}: {e}")

# Salva todas as previsões em CSV
if previsoes:
    df_previsoes = pd.DataFrame(previsoes)
    data_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    arquivo_saida = os.path.join(OUTPUT_DIR, f"previsoes_{data_str}.csv")
    df_previsoes.to_csv(arquivo_saida, index=False)
    print(f"\nTodas as previsões salvas em: {arquivo_saida}")
