# 🧠 EEG Mental State Classifier - Synapsee

Este projeto é uma ferramenta de Inteligência Artificial capaz de ler sinais cerebrais (EEG) e identificar se uma pessoa está **Relaxada**, **Concentrada** ou em estado **Neutro**. Além da classificação, a ferramenta gera um **Score de Engajamento** de 0 a 100%.

### 🛠️ Tecnologias Principais
* **Modelagem & XAI:** `XGBoost`, `Scikit-Learn`, `SHAP`, `SciPy`
* **Interface Visual:** `Streamlit`, `Plotly`, `Pandas`
* **Engenharia & Testes:** Arquitetura Modular (`src/`) e Testes Unitários (`pytest`)

---

## 🚀 Como Executar a Aplicação (Tutorial Passo a Passo)

Este tutorial foi feito para que qualquer pessoa, mesmo sem conhecimento em programação, consiga rodar o projeto.

### 1. Pré-requisitos
Você precisará do **Python** instalado em seu computador (versão 3.9 ou superior). Se não tiver, baixe em [python.org](https://www.python.org/).

### 2. Preparando o Ambiente
Abra o seu terminal (ou Prompt de Comando) na pasta deste projeto e digite os seguintes comandos:

```bash
# 1. Criar um ambiente isolado para não bagunçar seu PC
python -m venv venv

# 2. Ativar o ambiente (Windows)
venv\Scripts\activate

# 3. Instalar as bibliotecas necessárias
pip install -r requirements.txt
```

### 3. Rodando o Aplicativo
Com o ambiente ativado, basta digitar o comando abaixo:

```bash
streamlit run app.py
```

Uma aba abrirá automaticamente no seu navegador de internet com a interface do projeto.

### 🧪 Executando os Testes Automatizados
O projeto conta com uma cobertura inicial de testes unitários para o motor de feature engineering e avaliação de modelos. Para testar o código raiz:

1. Assegure-se de que o ambiente virtual está ativado.
2. Defina o mapeamento do projeto e execute os testes (exemplo para Windows PowerShell):
   ```powershell
   $env:PYTHONPATH="src"
   pytest tests/
   ```

---

## 📊 Como Usar a Ferramenta

1.  **Menu Lateral**: Clique em "Browse files" e escolha o arquivo de dados (Ex: `dataset/mental-state.csv`).
2.  **Processamento**: O sistema lerá os sinais, calculará as ondas Alpha/Beta e fará a predição.
3.  **Resultados**:
    *   **Cartão de Estado**: Mostrará seu estado mental atual com cores intuitivas.
    *   **Velocímetro de Foco**: Indicará seu nível de engajamento em tempo real.
    *   **Gráfico de Confiança**: Mostrará a certeza da Inteligência Artificial sobre o resultado.

---

## 📁 Arquitetura do Projeto
* `app.py`: O código central da interface visual interativa (Streamlit).
* `src/`: **Motor de Engenharia**. Scripts de processamento (`feature_pipeline.py`), treinamento cruzado de ML (`train_models.py`) e análise de negócio (`engagement_analysis.py`).
* `tests/`: Bateria de **testes unitários** (pytest) simulando comportamentos matemáticos cruciais das features. 
* `notebooks/EDA_Mental_State.ipynb`: Análise Exploratória de Dados científica detalhada visualmente.
* `RELATORIO_FINAL.md`: Documentação aprofundada contendo decisões metodológicas, performance dos modelos de M.L. testados e arquitetura (Leitura obrigatória para avaliadores!).
* `dataset/` e `models/`: Diretórios de armazenamento das amostras temporais brutas/processadas do EEG e do modelo final salvo em `.joblib`.

---
*Desenvolvido como parte do teste técnico para Data Scientist na Synapsee.*
