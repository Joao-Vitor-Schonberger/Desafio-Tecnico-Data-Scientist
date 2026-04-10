import pandas as pd
import numpy as np
import json
import os

def validate_hierarchy(trend):
    """Verifica se a hierarquia fisiológica do IEN é respeitada.
    
    Hierarquia esperada: Concentrado (2.0) > Relaxado (1.0) > Neutro (0.0)
    
    Justificativa neurocientífica:
    - Concentrado: Banda Beta elevada (atenção ativa) → IEN máximo.
    - Relaxado: Alpha elevado, mas Theta moderado → IEN intermediário.
    - Neutro: Desatenção passiva com Theta elevado, inflando o denominador
      da fórmula Beta/(Alpha+Theta), resultando no IEN mais baixo.
    """
    required_labels = [0.0, 1.0, 2.0]
    if not all(label in trend.index for label in required_labels):
        return False
    return trend[2.0] > trend[1.0] > trend[0.0]

def calculate_normalization_params(df):
    """Calcula percentis 0.05 e 0.95 do IEN_Global para normalização robusta."""
    ien_min = df['IEN_Global'].quantile(0.05)
    ien_max = df['IEN_Global'].quantile(0.95)
    return {
        'ien_min': float(ien_min),
        'ien_max': float(ien_max)
    }

def analyze_engagement(data_path, output_config='models/engagement_config.json'):
    print(f"Analisando métrica de engajamento: {data_path}")
    df = pd.read_csv(data_path)
    
    # 1. Validação da Tendência
    # Agrupar por Label e calcular a média do IEN_Global
    trend = df.groupby('Label')['IEN_Global'].mean().sort_index()
    
    label_names = {0.0: 'Neutro', 1.0: 'Relaxado', 2.0: 'Concentrado'}
    print("\nMédias de IEN por Estado Mental:")
    for label, value in trend.items():
        if label in label_names:
            print(f"{label_names[label]:12}: {value:.4f}")
    
    # Verificação de Coerência Fisiológica
    # Hierarquia esperada: Concentrado > Relaxado > Neutro
    # No estado Neutro (desatenção passiva), a potência Theta é significativamente
    # maior que nos outros estados, inflando o denominador do IEN e produzindo
    # o menor índice de engajamento — comportamento esperado pela literatura.
    is_valid = validate_hierarchy(trend)
    print(f"\nValidação de Hierarquia (Conc > Relax > Neutro): {'SUCESSO' if is_valid else 'FALHA'}")
    
    if is_valid:
        print("  >> Comportamento alinhado com a literatura neurocientifica.")
    
    # 2. Cálculo de Parâmetros de Normalização (Robust Scaling)
    params = calculate_normalization_params(df)
    
    config = {
        **params,
        'formula': 'min-max-clipping',
        'hierarchy_validated': bool(is_valid),
        'hierarchy_order': 'Concentrado > Relaxado > Neutro'
    }
    
    # Criar pasta models se não existir
    if not os.path.exists('models'):
        os.makedirs('models')
        
    with open(output_config, 'w') as f:
        json.dump(config, f, indent=4)
        
    print(f"\nConfiguração de Score salva em: {output_config}")
    print(f"Limites definidos: Min={params['ien_min']:.4f}, Max={params['ien_max']:.4f}")

if __name__ == "__main__":
    DATA_PATH = 'dataset/mental_state_featured.csv'
    if os.path.exists(DATA_PATH):
        analyze_engagement(DATA_PATH)
    else:
        print("Erro: Execute a Etapa 2 primeiro.")
