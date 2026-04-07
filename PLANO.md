# Plano de Projeto - Classificação de Estados Mentais (Versão 2.0)

## 1. Leitura do Problema (Atualizada)

### Objetivo Central
Construir um sistema end-to-end para classificar três estados mentais (relaxado, neutro, concentrado) usando sinais EEG processados. O projeto culminará em uma plataforma de inferência e um score contínuo de engajamento.

### Diagnóstico Técnico e Ajustes
Após inspeção inicial, detectamos que o dataset `mental-state.csv` está **embaralhado (shuffled)** e não possui identificadores de sujeito. 
*   **Decisão Técnica**: A estratégia original de Leave-One-Subject-Out (LOSO) foi descartada por inviabilidade técnica.
*   **Nova Estratégia**: Utilizaremos **Stratified K-Fold Cross-Validation** (5 folds) para garantir uma avaliação estatisticamente robusta das janelas de sinal.
*   **Foco**: O projeto focará na excelência da classificação por janelas e na extração de bandas de frequência (Alpha, Beta, Theta) a partir dos bins espectrais disponíveis.

---

## 2. Etapas e Estimativa de Tempo

### Etapa 0 — Setup e Inspeção (Concluída) `[~3h]`
- Configuração do ambiente e venv.
- Detecção de 115 duplicatas e remoção.
- Identificação do embaralhamento do dataset.

### Etapa 1 — Análise Exploratória (EDA) `[~4h]`
- Distribuição de classes (balanceadas).
- Correlação entre sensores (redundâncias).
- Visualização de features espectrais por estado mental.
- Projeção PCA para verificar separabilidade.

### Etapa 2 — Pipeline de Features e Bandas `[~6h]`
- **Agregação de Bandas**: Criar as bandas Delta, Theta, Alpha, Beta e Gamma somando os bins de frequência correspondentes (`freq_xxx_y`).
- **Cálculo do IEN**: Implementar o Índice de Engajamento Neurológico: `Beta / (Alpha + Theta)`.
- **Features Estatísticas**: Selecionar as melhores métricas temporais (médias, skewness, kurtosis) já presentes no dataset.

### Etapa 3 — Modelagem Comparativa `[~7h]`
- **Random Forest**: Para capturar relações não lineares e obter importância de features.
- **SVM (RBF)**: Como baseline robusto para alta dimensionalidade.
- **Validação**: Stratified K-Fold (k=5). Métricas: F1-score macro, Acurácia e Matriz de Confusão.

### Etapa 4 — Score de Engajamento e Relatório `[~5h]`
- Validar se `IEN(Concentrado) > IEN(Neutro) > IEN(Relaxado)`.
- Normalizar o score no intervalo [0, 1].

### Etapa 5 — Plataforma de Inferência `[~8h]`
- Interface Streamlit para upload de CSV.
- Exibição do estado mental e do score de engajamento.

---

## 3. Critérios de Sucesso
- **Acurácia**: ≥ 90% (esperada devido à sobreposição de janelas no embaralhamento).
- **Consistência**: O Score de Engajamento deve seguir a hierarquia esperada dos estados mentais.
- **Reprodutibilidade**: README claro e `requirements.txt` funcional.
