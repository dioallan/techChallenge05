# Previsão de Preço de Ações ITUB4 com LSTM

Projeto de Deep Learning utilizando LSTM (Long Short-Term Memory) para previsão do preço de fechamento da ação ITUB4.

## Objetivo

Desenvolver um modelo preditivo baseado em redes neurais recorrentes (LSTM) capaz de prever o preço futuro da ação ITUB4 com base em séries temporais históricas.

---

## Tecnologias Utilizadas

- Python 3.11
- TensorFlow / Keras
- Scikit-Learn
- Pandas / NumPy
- Matplotlib
- FastAPI
- Docker

---

## Arquitetura do Modelo

- Tipo: LSTM (Long Short-Term Memory)
- Camadas:
  - LSTM
  - Dropout
  - Dense
- Função de perda: Mean Squared Error (MSE)
- Otimizador: Adam

---

## Métricas de Avaliação

- MAE: 0.7241
- RMSE: 0.9519
- MAPE: (valor calculado no notebook)

---

## Estrutura do Projeto

lstm-prediction/
│
├── app.py
├── modelo_lstm_itub4.h5
├── scaler.pkl
├── requirements.txt
├── Dockerfile
├── README.md




---

##  Como Executar o Projeto

### Rodar Localmente

```bash
pip install -r requirements.txt
uvicorn app:app --reload

---

## Rodar Localmente

http://localhost:8000/docs

## Rodar com Docker

### Build da imagem

docker build -t petr4-api .

### Rodar Container
docker run -p 8000:8000 petr4-api

### Acessar

http://localhost:8000/docs


##Escalabilidade e Monitoramento

O projeto implementa:

Monitoramento de tempo de resposta via middleware FastAPI (header X-Process-Time)

Logs automáticos do Uvicorn

Script externo de monitoramento que:

Coleta dados reais via yfinance

Consome a API automaticamente

Armazena previsões em CSV com timestamp

Containerização via Docker para permitir escalabilidade horizontal