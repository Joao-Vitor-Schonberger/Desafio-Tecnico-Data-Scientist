import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, f1_score, accuracy_score
import os

def prepare_data(df):
    """Separa features e label, e aplica normalização."""
    X = df.drop('Label', axis=1)
    y = df['Label'].astype(int)
    
    # Normalização
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

def get_models():
    """Retorna os modelos a serem comparados."""
    return {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', probability=True, random_state=42),
        'XGBoost': XGBClassifier(eval_metric='mlogloss', random_state=42),
        'k-NN': KNeighborsClassifier(n_neighbors=5),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
    }

def train_and_compare(data_path, models_dir='models'):
    print(f"Carregando dados processados: {data_path}")
    df = pd.read_csv(data_path)
    
    X_scaled, y, scaler = prepare_data(df)
    models = get_models()
    
    # Configurando Validação Cruzada (Stratified K-Fold)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        'f1_macro': make_scorer(f1_score, average='macro'),
        'accuracy': 'accuracy'
    }
    
    results = []
    best_f1 = 0
    best_model_name = ""
    best_model_obj = None
    
    print("\nIniciando Treinamento e Validação Cruzada...")
    print("-" * 50)
    
    for name, model in models.items():
        cv_results = cross_validate(model, X_scaled, y, cv=skf, scoring=scoring)
        
        f1_mean = np.mean(cv_results['test_f1_macro'])
        acc_mean = np.mean(cv_results['test_accuracy'])
        
        results.append({
            'Modelo': name,
            'F1-Score Macro': f1_mean,
            'Acurácia': acc_mean
        })
        
        print(f"{name:20} | F1: {f1_mean:.4f} | Acc: {acc_mean:.4f}")
        
        if f1_mean > best_f1:
            best_f1 = f1_mean
            best_model_name = name
            best_model_obj = model
            
    print("-" * 50)
    print(f"Melhor Modelo: {best_model_name} (F1: {best_f1:.4f})")
    
    # Salvando o melhor modelo e o scaler
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    # Treinar o melhor modelo no dataset completo para produção
    best_model_obj.fit(X_scaled, y)
    
    joblib.dump(best_model_obj, os.path.join(models_dir, 'best_model.joblib'))
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    
    print(f"Modelo e Scaler salvos na pasta '{models_dir}'.")
    
    return pd.DataFrame(results)

if __name__ == "__main__":
    DATA_PATH = 'dataset/mental_state_featured.csv'
    if os.path.exists(DATA_PATH):
        results_df = train_and_compare(DATA_PATH)
        print("\nTabela Final de Resultados:")
        print(results_df.sort_values(by='F1-Score Macro', ascending=False))
    else:
        print(f"Erro: Arquivo {DATA_PATH} não encontrado. Execute a Etapa 2 primeiro.")
