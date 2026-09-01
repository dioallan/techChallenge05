# Previsão de Preço de Ações ITUB4 com LSTM

Projeto de Machine Learning Engineering utilizando uma rede neural **LSTM (Long Short-Term Memory)** para previsão do preço de fechamento da ação **ITUB4**.

O projeto contempla o ciclo de desenvolvimento e disponibilização de um modelo de Machine Learning, incluindo coleta de dados históricos, treinamento, avaliação, serialização, criação de API, containerização, deploy em produção e monitoramento automatizado.

---

## 1. Objetivo

Desenvolver um modelo preditivo capaz de estimar o próximo preço de fechamento da ação **ITUB4**, utilizando como entrada os últimos 60 preços de fechamento da série histórica.

A solução foi desenvolvida seguindo uma estratégia de **MLOps**, permitindo que o modelo treinado seja disponibilizado através de uma API e monitorado em produção.

---

## 2. Empresa Escolhida

A empresa escolhida para o projeto foi o **Itaú Unibanco**, utilizando o ticker:

```text
ITUB4.SA
```

Os dados históricos são obtidos através da biblioteca `yfinance`.

O treinamento utiliza dados históricos a partir de:

```text
2015-01-01
```

O atributo utilizado para a previsão é o **preço de fechamento (Close)**.

---

## 3. Tecnologias Utilizadas

* Python 3.11
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-Learn
* yfinance
* Joblib
* FastAPI
* Uvicorn
* Docker
* GitHub Actions
* Render

---

## 4. Modelo de Machine Learning

Foi utilizada uma rede neural recorrente do tipo **LSTM (Long Short-Term Memory)**, adequada para trabalhar com dados sequenciais e séries temporais.

### Configurações principais

* Janela temporal: **60 períodos**
* Divisão dos dados:

  * 80% treinamento
  * 20% teste
* Camada LSTM: **50 unidades**
* Camada de saída: `Dense(1)`
* Otimizador: **Adam**
* Função de perda: **Mean Squared Error (MSE)**
* Épocas: **10**
* Batch size: **32**

O modelo recebe uma sequência contendo os últimos 60 preços de fechamento e realiza a previsão do próximo valor.

---

## 5. Pré-processamento

Antes do treinamento, os dados são normalizados utilizando o `MinMaxScaler` do Scikit-Learn.

São criadas janelas temporais de 60 observações:

```text
[preço 1 ... preço 60] → previsão do preço 61
[preço 2 ... preço 61] → previsão do preço 62
...
```

O scaler utilizado no treinamento também é serializado para que a API possa aplicar exatamente o mesmo processo de transformação durante a inferência.

---

## 6. Avaliação do Modelo

Após o treinamento, o modelo realiza previsões utilizando o conjunto de teste.

Foram utilizadas as seguintes métricas:

* **MAE (Mean Absolute Error)**
* **RMSE (Root Mean Squared Error)**
* **MAPE (Mean Absolute Percentage Error)**

As métricas são calculadas no `train.py` após a previsão do conjunto de teste.

Os valores registrados no treinamento foram:

```text
MAE  : R$ 0.7241
RMSE : R$ 0.9519
MAPE : calculado durante o treinamento
```

Essas métricas permitem avaliar o erro das previsões do modelo em relação aos valores reais do conjunto de teste.

---

## 7. Serialização do Modelo

Após o treinamento, o modelo e o scaler são salvos para utilização posterior pela API.

### Modelo LSTM

```text
models/itub4_lstm.keras
```

### Scaler

```text
models/scaler_itub4.save
```

Dessa forma, a API não precisa realizar um novo treinamento a cada inicialização.

---

## 8. API FastAPI

Foi desenvolvida uma API utilizando **FastAPI** para disponibilizar o modelo em produção.

O principal endpoint é:

```text
POST /prever
```

Ele recebe uma lista contendo exatamente **60 preços de fechamento**.

### Exemplo de requisição

```json
{
  "valores": [
    81.1,
    80.9,
    81.0,
    "...",
    86.0,
    86.1
  ]
}
```

A API:

1. recebe os valores;
2. valida a quantidade de observações;
3. aplica o scaler;
4. prepara os dados no formato esperado pela LSTM;
5. executa a previsão;
6. converte o resultado novamente para a escala original;
7. retorna o preço previsto.

### Exemplo de resposta

```json
{
  "previsao_fechamento": 82.69547271728516
}
```

---

## 9. Documentação da API

A documentação interativa é disponibilizada pelo Swagger do FastAPI.

### API em produção

https://techchallenge05.onrender.com

### Swagger

https://techchallenge05.onrender.com/docs

Através do Swagger é possível visualizar os endpoints e executar uma requisição de previsão diretamente na API.

---

## 10. Containerização

A aplicação foi containerizada utilizando **Docker**.

O `Dockerfile` utiliza Python 3.11 e instala as dependências definidas no arquivo:

```text
requirements.txt
```

### Construção da imagem

```bash
docker build -t itub4-api .
```

### Execução do container

```bash
docker run -p 8000:8000 itub4-api
```

Após iniciar o container, a documentação pode ser acessada em:

```text
http://localhost:8000/docs
```

---

## 11. Deploy em Produção

A API foi disponibilizada em produção utilizando a plataforma **Render**.

O serviço executa a aplicação FastAPI dentro do ambiente containerizado.

A API disponível em produção é:

```text
https://techchallenge05.onrender.com
```

O endpoint de previsão utilizado no ambiente de produção é:

```text
POST https://techchallenge05.onrender.com/prever
```

---

## 12. Monitoramento em Produção

O monitoramento foi implementado utilizando **GitHub Actions**.

O workflow está localizado em:

```text
.github/workflows/monitoramento.yml
```

A execução automática ocorre a cada **10 minutos**.

Também é possível executar o workflow manualmente através do GitHub Actions.

### Estratégia de monitoramento

O workflow realiza uma chamada real ao endpoint `/prever` da API em produção, utilizando uma entrada válida contendo 60 valores.

O comando utilizado é baseado em:

```bash
curl -f -X POST
```

O parâmetro `-f` faz com que a execução seja considerada uma falha quando a API retorna um erro HTTP.

Dessa forma, o monitoramento verifica não apenas se o servidor está disponível, mas também se o endpoint de inferência está respondendo corretamente.

### Fluxo

```text
GitHub Actions
      ↓
A cada 10 minutos
      ↓
POST /prever
      ↓
API FastAPI em produção
      ↓
Modelo LSTM ITUB4
      ↓
Previsão
      ↓
HTTP 200
```

Além disso, a API possui um middleware que registra o tempo de processamento das requisições e disponibiliza essa informação através do header:

```text
X-Process-Time
```

---

## 13. Estrutura do Projeto

```text
techChallenge05/
│
├── .github/
│   └── workflows/
│       └── monitoramento.yml
│
├── api/
│   └── app.py
│
├── models/
│   ├── itub4_lstm.keras
│   └── scaler_itub4.save
│
├── notebook/
│   └── 01_lstm_training.ipynb
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── train.py
└── teste.py
```

---

## 14. Como Executar Localmente

### Criar ambiente virtual

```bash
python -m venv .venv
```

### Ativar o ambiente virtual no Windows

```bash
.venv\Scripts\activate
```

### Instalar as dependências

```bash
pip install -r requirements.txt
```

### Executar a API

```bash
uvicorn api.app:app --reload
```

A documentação estará disponível em:

```text
http://localhost:8000/docs
```

---

## 15. Treinamento do Modelo

Para realizar um novo treinamento:

```bash
python train.py
```

O script:

1. coleta os dados históricos de ITUB4;
2. seleciona os preços de fechamento;
3. normaliza os dados;
4. cria as janelas temporais;
5. divide os dados em treinamento e teste;
6. cria e treina a rede LSTM;
7. realiza previsões no conjunto de teste;
8. calcula MAE, RMSE e MAPE;
9. salva o modelo;
10. salva o scaler.

---

## 16. Estratégia de MLOps

A estratégia implementada neste projeto pode ser resumida da seguinte forma:

```text
Dados históricos
       ↓
Pré-processamento
       ↓
Treinamento LSTM
       ↓
Avaliação
       ↓
Serialização
       ↓
API FastAPI
       ↓
Docker
       ↓
Deploy no Render
       ↓
Monitoramento com GitHub Actions
       ↓
Inferência periódica em produção
```

O objetivo é demonstrar não somente a criação do modelo de Machine Learning, mas também sua disponibilização e acompanhamento em um ambiente de produção.

---

## 17. Repositório

O código-fonte, modelo serializado, scaler, dependências, Dockerfile, workflow de monitoramento e documentação estão disponíveis no repositório GitHub do projeto.

---

## 18. Conclusão

O projeto implementa uma solução completa para previsão do preço de fechamento da ação ITUB4 utilizando LSTM.

Foram contempladas as principais etapas de uma estratégia de Machine Learning Engineering/MLOps:

* coleta de dados históricos;
* preparação dos dados;
* treinamento do modelo;
* avaliação através de métricas;
* serialização;
* criação de API;
* containerização;
* deploy em produção;
* monitoramento automatizado da inferência;
* documentação do projeto.
