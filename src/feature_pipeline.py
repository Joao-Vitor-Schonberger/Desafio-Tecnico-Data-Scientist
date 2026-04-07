import pandas as pd
import numpy as np
from scipy import stats
import os

def clean_duplicates(df):
    """Remove duplicatas do DataFrame."""
    return df.drop_duplicates()

def remove_outliers(df, mean_cols):
    """Remove outliers usando Z-score > 3 nas colunas especificadas."""
    z_scores = np.abs(stats.zscore(df[mean_cols]))
    return df[(z_scores < 3).all(axis=1)]

def aggregate_frequency_bands(df, s):
    """Agrega bins de frequência em bandas (Delta, Theta, Alpha, Beta, Gamma)."""
    # Localizar colunas de frequência para o sensor s
    freq_cols = [c for c in df.columns if f'freq_' in c and c.endswith(f'_{s}') and 'lag' not in c]
    
    # Criar bandas agregadas (Médias dos bins correspondentes)
    df[f'Delta_{s}'] = df[freq_cols[0:5]].mean(axis=1)
    df[f'Theta_{s}'] = df[freq_cols[5:12]].mean(axis=1)
    df[f'Alpha_{s}'] = df[freq_cols[12:20]].mean(axis=1)
    df[f'Beta_{s}'] = df[freq_cols[20:45]].mean(axis=1)
    df[f'Gamma_{s}'] = df[freq_cols[45:]].mean(axis=1)
    return df

def calculate_ien_per_sensor(df, s):
    """Calcula o Índice de Engajamento Neurológico (IEN) por sensor."""
    # Fórmula: Beta / (Alpha + Theta)
    # Adicionamos um pequeno epsilon para evitar divisão por zero
    epsilon = 1e-6
    return df[f'Beta_{s}'] / (df[f'Alpha_{s}'] + df[f'Theta_{s}'] + epsilon)

def run_pipeline(input_path, output_path):
    print(f"Iniciando pipeline de features: {input_path}")
    
    # 1. Carregamento
    df = pd.read_csv(input_path)
    initial_rows = len(df)
    
    # 2. Limpeza de Duplicatas
    df = clean_duplicates(df)
    
    # 3. Remoção de Outliers (Z-score > 3 nas colunas de média)
    mean_cols = [c for c in df.columns if 'mean' in c and 'lag' not in c and 'q' not in c]
    df = remove_outliers(df, mean_cols)
    
    print(f"Limpeza concluída. Linhas removidas: {initial_rows - len(df)}")
    
    # 4. Agregação de Bandas de Frequência e Cálculo de IEN
    processed_features = []
    
    for s in range(4):
        df = aggregate_frequency_bands(df, s)
        df[f'IEN_{s}'] = calculate_ien_per_sensor(df, s)
        processed_features.extend([f'Delta_{s}', f'Theta_{s}', f'Alpha_{s}', f'Beta_{s}', f'Gamma_{s}', f'IEN_{s}'])

    # 6. Feature de Engenharia Global (Média do IEN entre todos os sensores)
    df['IEN_Global'] = df[[f'IEN_{i}' for i in range(4)]].mean(axis=1)
    processed_features.append('IEN_Global')
    
    # 7. Manter as features estatísticas temporais originais importantes
    original_stats = [c for c in df.columns if ('mean' in c or 'std' in c) and 'lag' not in c and 'freq' not in c and 'q' not in c]
    
    # Dataset Final
    final_cols = processed_features + original_stats + ['Label']
    df_final = df[final_cols]
    
    # 8. Exportação
    df_final.to_csv(output_path, index=False)
    print(f"Pipeline finalizado com sucesso! Arquivo salvo em: {output_path}")
    print(f"Total de features geradas: {len(df_final.columns) - 1}")

if __name__ == "__main__":
    INPUT = 'dataset/mental-state.csv'
    OUTPUT = 'dataset/mental_state_featured.csv'
    
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
        
    run_pipeline(INPUT, OUTPUT)
