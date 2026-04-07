import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.graph_objects as go
import os
from scipy import stats
import shap

# Configurações da Página
st.set_page_config(page_title="Synapsee - EEG Mental State", layout="wide")

# Funções de Suporte
@st.cache_resource
def load_assets():
    model = joblib.load('models/best_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    with open('models/engagement_config.json', 'r') as f:
        config = json.load(f)
    return model, scaler, config

def process_features(df):
    """Aplica a mesma lógica da Etapa 2 para novos dados de forma otimizada"""
    new_features = {}
    epsilon = 1e-6
    
    # Bandas
    for s in range(4):
        freq_cols = [c for c in df.columns if f'freq_' in c and c.endswith(f'_{s}') and 'lag' not in c]
        if not freq_cols: continue
        
        delta = df[freq_cols[0:5]].mean(axis=1)
        theta = df[freq_cols[5:12]].mean(axis=1)
        alpha = df[freq_cols[12:20]].mean(axis=1)
        beta = df[freq_cols[20:45]].mean(axis=1)
        gamma = df[freq_cols[45:]].mean(axis=1)
        
        ien = beta / (alpha + theta + epsilon)
        
        new_features[f'Delta_{s}'] = delta
        new_features[f'Theta_{s}'] = theta
        new_features[f'Alpha_{s}'] = alpha
        new_features[f'Beta_{s}'] = beta
        new_features[f'Gamma_{s}'] = gamma
        new_features[f'IEN_{s}'] = ien

    df_features = pd.DataFrame(new_features)
    df_features['IEN_Global'] = df_features[[f'IEN_{i}' for i in range(4)]].mean(axis=1)
    
    original_stats = [c for c in df.columns if ('mean' in c or 'std' in c) and 'lag' not in c and 'freq' not in c and 'q' not in c]
    df_original = df[original_stats]
    
    df_final = pd.concat([df_features, df_original], axis=1)
    return df_final, df_features['IEN_Global']

# Interface Principal
st.title("🧠 Synapsee Mental State Classifier")
st.markdown("---")

# Verificação de Ativos
if not os.path.exists('models/best_model.joblib'):
    st.error("Modelos não encontrados. Execute as etapas de treino primeiro.")
    st.stop()

model, scaler, eng_config = load_assets()

# Sidebar para Upload
st.sidebar.header("Entrada de Dados")
uploaded_file = st.sidebar.file_uploader("Suba o CSV com sinais EEG", type="csv")

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    
    # Se o arquivo já tiver o Label, vamos remover para predição
    if 'Label' in data.columns:
        real_labels = data['Label']
        data_to_pred = data.drop('Label', axis=1)
    else:
        data_to_pred = data
        real_labels = None

    try:
        # Processamento
        X_processed, ien_values = process_features(data_to_pred)
        X_scaled = scaler.transform(X_processed)
        
        # Predições
        preds = model.predict(X_scaled)
        probs = model.predict_proba(X_scaled)
        
        # SHAP - Calculando a explicação para a média da sessão
        # Usar a média do usuário garante que o app não trave renderizando 2000 linhas
        explainer = shap.TreeExplainer(model)
        mean_x_unscaled = X_processed.mean(axis=0).to_frame().T
        mean_x_scaled = scaler.transform(mean_x_unscaled)
        pred_class_mean = int(model.predict(mean_x_scaled)[0])
        
        shap_values = explainer.shap_values(mean_x_scaled)
        
        # Tratamento seguro para formato de saída do XGBoost Multiclasse
        if isinstance(shap_values, list):
            shap_vals_for_class = shap_values[pred_class_mean][0]
        elif len(np.array(shap_values).shape) == 3:
            shap_vals_for_class = np.array(shap_values)[0, :, pred_class_mean]
        else:
            shap_vals_for_class = np.array(shap_values)[0]
            
        feature_names = X_processed.columns
        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': shap_vals_for_class})
        shap_df['Abs Value'] = shap_df['SHAP Value'].abs()
        # Pegar as 10 features mais importantes e ordenar para o gráfico horizontal
        shap_df = shap_df.sort_values(by='Abs Value', ascending=False).head(10)
        shap_df = shap_df.sort_values(by='Abs Value', ascending=True)

        # Mapeamento
        label_map = {0: 'Neutro', 1: 'Relaxado', 2: 'Concentrado'}
        
        # Layout de Resultados - Linha 1
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Estado Mental Dominante")
            most_freq_pred = stats.mode(preds, keepdims=False)[0] if len(preds) > 1 else preds[0]
            label_name = label_map[int(most_freq_pred)]
            
            color = "#3498db" if label_name == 'Neutro' else "#2ecc71" if label_name == 'Relaxado' else "#e74c3c"
            st.markdown(f"<div style='padding:20px; border-radius:10px; background-color:{color}; color:white; text-align:center; font-size:30px; font-weight:bold;'>{label_name}</div>", unsafe_allow_html=True)
            
            st.write("")
            st.subheader("Confiança do Modelo")
            avg_probs = np.mean(probs, axis=0)
            fig_prob = go.Figure(go.Bar(
                x=list(label_map.values()),
                y=avg_probs,
                marker_color=["#3498db", "#2ecc71", "#e74c3c"]
            ))
            fig_prob.update_layout(height=250, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_prob, use_container_width=True)

        with col2:
            st.subheader("Score de Engajamento")
            avg_ien = np.mean(ien_values)
            ien_min = eng_config['ien_min']
            ien_max = eng_config['ien_max']
            
            engagement_score = (avg_ien - ien_min) / (ien_max - ien_min)
            engagement_score = np.clip(engagement_score, 0, 1)
            
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = engagement_score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Nível de Foco (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#2c3e50"},
                    'steps': [
                        {'range': [0, 30], 'color': "#ecf0f1"},
                        {'range': [30, 70], 'color': "#bdc3c7"},
                        {'range': [70, 100], 'color': "#95a5a6"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            if engagement_score < 0.3:
                st.info("💡 Interpretação: Estado de descompressão ou baixa demanda cognitiva.")
            elif engagement_score < 0.7:
                st.info("💡 Interpretação: Atenção equilibrada e monitoramento estável.")
            else:
                st.warning("🔥 Interpretação: Alto engajamento. Provável estado de Flow ou Concentração Intensa.")

        # Layout de Resultados - Linha 2 (Explicabilidade SHAP)
        st.markdown("---")
        st.subheader("🧠 Por que a IA tomou essa decisão? (Explainable AI)")
        st.markdown(f"O gráfico abaixo revela a **'caixa preta'** do algoritmo. Ele mostra as 10 características neurais que mais influenciaram a Inteligência Artificial a classificar seu estado médio como **{label_map[pred_class_mean]}**.")
        
        fig_shap = go.Figure(go.Bar(
            x=shap_df['SHAP Value'],
            y=shap_df['Feature'],
            orientation='h',
            marker_color=['#2ecc71' if val > 0 else '#e74c3c' for val in shap_df['SHAP Value']],
            text=[f"{val:+.3f}" for val in shap_df['SHAP Value']],
            textposition='auto'
        ))
        fig_shap.update_layout(
            title="Contribuição das Features (SHAP Values)",
            xaxis_title="Impacto na Decisão (Verde empurra a favor da classificação, Vermelho joga contra)",
            yaxis_title="Feature Cerebral",
            height=400,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_shap, use_container_width=True)

        with st.expander("Ver dados processados brutos"):
            st.dataframe(X_processed.head(10))

    except Exception as e:
        st.error(f"Erro ao processar o arquivo: {e}")
        st.info("Certifique-se de que o CSV possui as colunas originais de frequência e estatísticas.")

else:
    st.info("Aguardando upload de dados... Use o menu lateral.")
    st.image("https://images.unsplash.com/photo-1559757175-5700dde675bc?ixlib=rb-1.2.1&auto=format&fit=crop&w=1350&q=80", caption="Análise Neural via EEG")
    
    st.markdown("""
    ### Como usar:
    1. Prepare um arquivo CSV com as 988 features originais.
    2. Arraste para o campo de upload.
    3. O sistema classificará instantaneamente o seu estado mental.
    """)
