	**Synapsee**

---

## Contexto

Na Synapsee, utilizamos dados neurais e de eye-tracking para compreender emoções e atenção de usuários expostos a estímulos visuais — e transformamos isso em inteligência para otimizar criativos de grandes empresas.

Neste desafio, você vai construir um sistema de classificação de estados mentais a partir de sinais de EEG, do pré-processamento até uma plataforma funcional de inferência.

---

## Dataset

**EEG Brainwave Dataset: Mental State**
https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-mental-state

- Sinais EEG de 2 participantes com estados mentais rotulados: relaxado, neutro e concentrado
- Dataset compacto e bem estruturado

> Sinta-se livre para usar outro dataset de EEG ou fisiológico que julgue mais interessante — só justifique a escolha no seu relatório.

---

## O que você deve entregar

### 1. PLANO.md — entregue antes de escrever qualquer código

Antes de começar, escreva um documento curto com:

- Sua leitura do problema e o que você pretende resolver
- As etapas que você vai executar, em ordem, com estimativa de tempo para cada uma
- Os critérios que você mesmo vai usar para saber se o resultado é bom

Envie este arquivo para nós antes de começar a desenvolver. Ele é parte da avaliação.

---

### 2. Análise exploratória

Notebook com distribuição dos sinais, correlações entre canais e diferenças entre estados mentais. Mostre que você entende os dados antes de modelar.

### 3. Pipeline de features

Extraia features relevantes dos sinais — por exemplo: bandas de frequência (alpha, beta, theta), estatísticas temporais, densidade espectral de potência (PSD). Justifique cada escolha.

### 4. Modelo preditivo

Implemente e compare pelo menos **duas abordagens**. Use métricas adequadas e discuta overfitting e capacidade de generalização.

### 5. Score de engajamento

Proponha e calcule um score contínuo que represente o nível de engajamento do participante. Não precisa ser complexo — precisa ser coerente e bem explicado.

### 6. Plataforma de inferência

Uma interface simples (Streamlit ou similar) onde o usuário faz upload de um CSV de sinais e recebe:

- A classificação do estado mental
- O score de engajamento

### 7. Relatório final

Um documento em Markdown com o que funcionou, o que não funcionou e o que você faria diferente com mais tempo ou dados.

---

## Regras

- **Use IA à vontade** — Claude Code, Copilot, Cursor, ChatGPT. Encorajamos o uso dessas ferramentas.
- **Envie o PLANO.md antes de começar a codar.** Ele é parte da avaliação.
- **Suba tudo em um repositório Git** com um README claro de como rodar o projeto.
- **Você apresentará os resultados ao vivo.** Faremos perguntas sobre suas decisões de modelagem, os trade-offs envolvidos e o que você aprendeu no processo.
- **Prazo:** combinado individualmente

---

## Critérios de avaliação

- Qualidade do planejamento e aderência ao plano (PLANO.md vs. entrega)
- Rigor na análise exploratória e justificativa das features
- Comparação honesta entre abordagens de modelagem
- Coerência do score de engajamento
- Funcionamento da plataforma de inferência
- Clareza do raciocínio — verificada ao vivo
- ⁠Se respeitou o próprio prazo dado