# Relatório Final - Projeto de Classificação Mental (EEG)

Este relatório detalha as decisões arquiteturais, metodológicas e os resultados obtidos no desenvolvimento da plataforma Synapsee EEG.

---

## 1. Tratamento de Dados e Limitações

### 1.1 Diagnóstico do Dataset
Durante a fase de inspeção (Etapa 0), identificamos que o dataset público fornecido estava **completamente embaralhado (shuffled)** e carecia de identificadores de participante (`Subject_ID`).

*   **Impacto direto**: A estratégia de validação original — Leave-One-Subject-Out (LOSO), que é o padrão-ouro em BCI — foi tecnicamente inviabilizada. Sem saber qual janela pertence a qual sujeito, qualquer split pode misturar janelas do mesmo participante entre treino e teste, gerando **data leakage implícito**.
*   **Mitigação adotada**: Implementamos **Validação Cruzada Estratificada (K-Fold, k=5)** para garantir que cada predição fosse testada em subconjuntos com proporção balanceada das três classes.

### 1.2 Limpeza
*   Remoção de **115 duplicatas** exatas detectadas no dataset bruto.
*   Remoção de outliers via **Z-score > 3** nas colunas de média dos sensores, eliminando janelas com provável falha de sensor ou artefato de movimento.

---

## 2. Engenharia de Features (Neurociência)

Para elevar o modelo além do tratamento genérico de dados tabulares, transformamos os 72 bins de frequência brutos em **bandas fisiológicas reais**:

| Banda | Faixa aprox. | Significado Fisiológico |
| :--- | :--- | :--- |
| **Delta** | 0.5–4 Hz | Sono profundo / artefatos |
| **Theta** | 4–8 Hz | Relaxamento profundo, memória, desatenção passiva |
| **Alpha** | 8–13 Hz | Relaxamento alerta (inversamente proporcional ao engajamento) |
| **Beta** | 13–30 Hz | Concentração ativa, atenção sustentada |
| **Gamma** | 30–45 Hz | Processamento cognitivo intenso |

A agregação confirmou hipóteses da literatura: a potência **Alpha cai durante a concentração** em todos os 4 sensores, e a potência **Theta é significativamente mais alta no estado Neutro** (desatenção passiva), validando o embasamento neurocientífico do pipeline.

---

## 3. Performance da Modelagem

### 3.1 Torneio de Modelos
Testamos 6 modelos sob as mesmas condições de normalização (StandardScaler) e validação (Stratified K-Fold, k=5):

| Modelo | Acurácia (K-Fold) | F1-Score (Macro) |
| :--- | :--- | :--- |
| **XGBoost** | **97.53%** | **0.9765** |
| Extra Trees | 97.05% | 0.9727 |
| Random Forest | 96.86% | 0.9708 |
| SVM (RBF) | 96.06% | 0.9631 |
| k-NN | 94.16% | 0.9463 |
| Logistic Regression | 93.26% | 0.9373 |

A vitória do **XGBoost** justifica-se por sua capacidade de lidar com alta dimensionalidade, capturar interações complexas entre bandas de frequência de diferentes sensores e realizar regularização interna (L1/L2), reduzindo overfitting.

### 3.2 Discussão sobre Overfitting e Generalização

A acurácia obtida (~97.5%) é alta, mas merece uma análise cautelosa:

*   **Risco de data leakage**: Como o dataset original está embaralhado sem `Subject_ID`, é possível que janelas temporalmente adjacentes do mesmo sujeito apareçam tanto no treino quanto no teste. Janelas de EEG de um mesmo participante tendem a ter alta correlação, o que inflaciona métricas de validação cruzada. Em um cenário real com novos sujeitos, a acurácia esperada seria provavelmente **10–15 pontos percentuais menor**.
*   **Mitigação possível**: Caso IDs de sujeito estivessem disponíveis, utilizaríamos GroupKFold ou LOSO para uma estimativa mais honesta de generalização cross-subject.
*   **Evidência positiva**: Modelos lineares (Logistic Regression) alcançaram 93.26%, indicando que o sinal discriminativo é forte e não depende exclusivamente de overfitting de modelos complexos. O gap entre o modelo mais simples e o XGBoost (~4 pontos) sugere ganho real de modelagem não-linear.
*   **Observação sobre os top-3**: A proximidade entre XGBoost (97.53%), Extra Trees (97.05%) e Random Forest (96.86%) sugere que o teto de performance está sendo atingido e que a diferença entre eles é marginal — o benefício do XGBoost é sua robustez, não uma vantagem drástica.

---

## 4. O Score de Engajamento

### 4.1 Fórmula e Fundamentação
O score contínuo (0–100%) foi baseado no **Índice de Engajamento Neurológico (IEN)**, também conhecido como **Índice de Pope**:

```
IEN = Beta / (Alpha + Theta)
```

### 4.2 Validação da Hierarquia

A validação confirmou a seguinte hierarquia fisiológica:

| Estado Mental | IEN Médio | Interpretação |
| :--- | :--- | :--- |
| **Concentrado** | 0.3414 | Beta elevado → maior engajamento |
| **Relaxado** | 0.2437 | Alpha moderado, Theta baixo → engajamento intermediário |
| **Neutro** | 0.2252 | Desatenção passiva, Theta muito alto → menor engajamento |

**Hierarquia validada: Concentrado > Relaxado > Neutro** ✅

É importante notar que a ordem `Relaxado > Neutro` é fisiologicamente esperada: no estado Neutro (apatia/desatenção passiva), a potência Theta é significativamente mais alta que nos outros estados em 3 dos 4 sensores. Como Theta está no denominador da fórmula, isso infla o denominador e reduz o IEN abaixo do estado Relaxado.

### 4.3 Normalização
O score final é normalizado via **min-max com clipping usando percentis robustos** (P5 e P95), evitando que outliers extremos distorçam a escala. O resultado é um valor contínuo entre 0% e 100%.

---

## 5. O que Não Funcionou

### 5.1 Hierarquia Inicial Incorreta
Na primeira versão do score, assumimos a hierarquia `Concentrado > Neutro > Relaxado`, onde Neutro seria um estado intermediário entre foco e relaxamento. Os dados **refutaram essa hipótese**: o estado Neutro apresenta as menores médias de IEN, não as intermediárias. Essa descoberta foi corrigida ao investigar as bandas por sensor, revelando que o Theta no estado Neutro é consistentemente o mais alto.

### 5.2 Validação Leave-One-Subject-Out (LOSO)
Planejamos inicialmente usar LOSO — o padrão em pesquisa de BCI — para avaliar a capacidade real de generalização para novos sujeitos. Porém, a ausência de identificadores no dataset inviabilizou completamente essa abordagem.

### 5.3 Sensores Periféricos (0 e 3)
Os sensores 0 (TP9) e 3 (TP10), localizados na região temporal posterior, apresentaram baixa capacidade discriminativa no IEN — o spread `Concentrado - Relaxado` foi quase zero nesses canais. Os sensores frontais (1: AF7, 2: AF8) carregaram a maior parte do poder discriminativo, o que é consistente com o papel do córtex pré-frontal na atenção executiva.

---

## 6. Conclusões e Melhorias Futuras

O projeto entrega uma ferramenta funcional que classifica estados mentais com alta acurácia (~97.5%) e fornece um score de engajamento fisiologicamente fundamentado e validado.

### O que faria diferente com mais tempo ou dados:
1.  **Redes Neurais Recorrentes (LSTM)**: Para capturar dinâmicas temporais entre janelas consecutivas, tratando o EEG como série temporal e não como amostras independentes.
2.  **Validação cross-subject real**: Com dados contendo IDs de participante, implementar LOSO para obter métricas de generalização honestas.
3.  **Remoção de artefatos oculares (EOG)**: Aplicação de ICA (Independent Component Analysis) no sinal bruto antes da extração de features, removendo contaminação de piscadas e movimentos oculares.
4.  **Seleção de sensores**: Dado que os sensores temporais (0, 3) contribuíram pouco, testar pipelines usando apenas os frontais (AF7, AF8) para reduzir dimensionalidade sem perda significativa de performance.
5.  **Calibração de probabilidades**: Aplicar Platt Scaling ou isotonic regression para que as probabilidades do XGBoost sejam verdadeiramente calibradas, melhorando a confiabilidade do score exibido ao usuário.

