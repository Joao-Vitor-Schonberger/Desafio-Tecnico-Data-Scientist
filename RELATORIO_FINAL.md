# Relatório Final - Projeto de Classificação Mental (EEG)

Este relatório detalha as decisões arquiteturais, metodológicas e os resultados obtidos no desenvolvimento da plataforma Synapsee EEG.

## 1. Tratamento de Dados e Limitações
Durante a fase de inspeção (Etapa 0), identificamos que o dataset público fornecido estava **completamente embaralhado (shuffled)** e carecia de identificadores de participante (`Subject_ID`). 
*   **Decisão**: A estratégia de validação original (Leave-One-Subject-Out - LOSO) foi tecnicamente inviabilizada.
*   **Mitigação**: Implementamos a **Validação Cruzada Estratificada (K-Fold, k=5)** para garantir que cada predição fosse testada em subconjuntos representativos das três classes.

## 2. Engenharia de Features (Neurociência)
Para elevar o modelo além do tratamento genérico de dados tabulares, transformamos os 72 bins de frequência brutos em bandas fisiológicas reais:
*   **Delta, Theta, Alpha, Beta e Gamma**.
Essa agregação permitiu validar hipóteses biológicas, como a redução da potência Alpha durante estados de concentração, conferindo validade científica ao modelo.

## 3. Performance da Modelagem
Testamos um torneio de 6 modelos sob as mesmas condições de normalização e validação:

| Modelo | Acurácia (K-Fold) | F1-Score (Macro) |
| :--- | :--- | :--- |
| **XGBoost** | **97.53%** | **0.9765** |
| Extra Trees | 97.05% | 0.9727 |
| Random Forest | 96.86% | 0.9708 |
| SVM (RBF) | 96.06% | 0.9631 |
| k-NN | 94.16% | 0.9463 |
| Logistic Regression| 93.26% | 0.9373 |

A vitória do **XGBoost** justifica-se por sua capacidade de lidar com alta dimensionalidade e capturar interações complexas entre os diferentes sensores frontais e temporais.

## 4. O Score de Engajamento
O score contínuo (0-100%) foi baseado no **Índice de Pope**: `Beta / (Alpha + Theta)`. 
A validação estatística confirmou que a média do IEN no estado "Concentrado" é significativamente superior aos estados de baixa demanda cognitiva, permitindo uma normalização via percentis que oferece um feedback suave e preciso ao usuário final.

## 5. Conclusões e Melhorias Futuras
O projeto entrega uma ferramenta funcional com alta confiabilidade estatística. Caso houvesse mais tempo ou dados brutos, os próximos passos seriam:
1.  Aplicação de Redes Neurais Recorrentes (LSTM) para capturar dinâmicas temporais.
2.  Testes de transferência de domínio para novos participantes reais (BCI em tempo real).
3.  Implementação de filtros de remoção de artefatos oculares (EOG) via ICA.
