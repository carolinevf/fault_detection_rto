"""
Funções para modelagem e avaliação de modelos.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans, AgglomerativeClustering
from rto_analysis.config import MODELS_DIR, RANDOM_STATE

logger = logging.getLogger(__name__)

def train_models(X, y, models_to_train=None):
    """
    Treina múltiplos modelos de classificação.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features para treinamento
    y : pandas.Series
        Target para treinamento
    models_to_train : list, opcional
        Lista de modelos para treinar. Se None, treina todos os modelos disponíveis.
        
    Returns:
    --------
    dict
        Dicionário contendo os modelos treinados
    """
    logger.info("Iniciando treinamento de modelos")
    
    # Definir modelos disponíveis
    available_models = {
        'logistic_regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        'random_forest': RandomForestClassifier(random_state=RANDOM_STATE),
        'knn': KNeighborsClassifier(weights='distance')
    }
    
    # Selecionar modelos para treinar
    if models_to_train is None:
        models_to_train = available_models.keys()
    
    # Treinar modelos
    trained_models = {}
    for model_name in models_to_train:
        if model_name not in available_models:
            logger.warning(f"Modelo {model_name} não disponível, pulando")
            continue
        
        logger.info(f"Treinando modelo: {model_name}")
        model = available_models[model_name]
        model.fit(X, y)
        trained_models[model_name] = model
        
        # Salvar modelo
        model_path = MODELS_DIR / f"{model_name}.joblib"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        logger.info(f"Modelo {model_name} salvo em {model_path}")
    
    return trained_models

def evaluate_models(models, X, y, save_path=None):
    """
    Avalia modelos de classificação.
    
    Parameters:
    -----------
    models : dict
        Dicionário contendo os modelos treinados
    X : pandas.DataFrame
        Features para avaliação
    y : pandas.Series
        Target para avaliação
    save_path : str, opcional
        Caminho para salvar as figuras de avaliação
        
    Returns:
    --------
    dict
        Dicionário contendo métricas de avaliação para cada modelo
    """
    logger.info("Avaliando modelos")
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
    
    results = {}
    for model_name, model in models.items():
        logger.info(f"Avaliando modelo: {model_name}")
        
        # Fazer predições
        y_pred = model.predict(X)
        
        # Calcular métricas
        report = classification_report(y, y_pred, output_dict=True)
        cm = confusion_matrix(y, y_pred)
        
        # Armazenar resultados
        results[model_name] = {
            'classification_report': report,
            'confusion_matrix': cm
        }
        
        # Plotar matriz de confusão
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Não Converge', 'Converge'],
                    yticklabels=['Não Converge', 'Converge'])
        plt.xlabel('Predito')
        plt.ylabel('Real')
        plt.title(f'Matriz de Confusão - {model_name}')
        
        if save_path:
            plt.savefig(save_path / f"confusion_matrix_{model_name}.png")
            logger.info(f"Matriz de confusão salva em {save_path}/confusion_matrix_{model_name}.png")
        plt.close()
        
        # Plotar curva ROC se o modelo tiver predict_proba
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)[:, 1]
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel('Taxa de Falsos Positivos')
            plt.ylabel('Taxa de Verdadeiros Positivos')
            plt.title(f'Curva ROC - {model_name}')
            plt.legend(loc='lower right')
            
            if save_path:
                plt.savefig(save_path / f"roc_curve_{model_name}.png")
                logger.info(f"Curva ROC salva em {save_path}/roc_curve_{model_name}.png")
            plt.close()
            
            # Adicionar AUC aos resultados
            results[model_name]['roc_auc'] = roc_auc
            
            # Plotar curva Precision-Recall
            precision, recall, _ = precision_recall_curve(y, y_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Curva Precision-Recall - {model_name}')
            
            if save_path:
                plt.savefig(save_path / f"pr_curve_{model_name}.png")
                logger.info(f"Curva Precision-Recall salva em {save_path}/pr_curve_{model_name}.png")
            plt.close()
    
    return results

def optimize_hyperparameters(X, y, model_name, param_grid, cv=5):
    """
    Otimiza hiperparâmetros de um modelo.
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Features para treinamento
    y : pandas.Series
        Target para treinamento
    model_name : str
        Nome do modelo para otimizar
    param_grid : dict
        Grade de parâmetros para otimização
    cv : int
        Número de folds para validação cruzada
        
    Returns:
    --------
    sklearn.model_selection.GridSearchCV
        Objeto GridSearchCV com o melhor modelo
    """
    logger.info(f"Otimizando hiperparâmetros para o modelo {model_name}")
    
    # Definir modelo base
    if model_name == 'logistic_regression':
        model = LogisticRegression(random_state=RANDOM_STATE)
    elif model_name == 'random_forest':
        model = RandomForestClassifier(random_state=RANDOM_STATE)
    elif model_name == 'knn':
        model = KNeighborsClassifier()
    else:
        raise ValueError(f"Modelo {model_name} não suportado para otimização")
    
    # Executar GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
    grid_search.fit(X, y)
    
    logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
    logger.info(f"Melhor score: {grid_search.best_score_:.4f}")
    
    # Salvar modelo otimizado
    model_path = MODELS_DIR / f"{model_name}_optimized.joblib"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid_search.best_estimator_, model_path)
    logger.info(f"Modelo otimizado salvo em {model_path}")
