import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

def setup_logging(log_file='rto_analysis.log'):
    """
    Configura o logging para o projeto.
    
    Parameters:
    -----------
    log_file : str
        Nome do arquivo de log
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=log_file
    )

def apply_savgol_filter(df, col, n_window=5, degree=2):
    """
    Aplica filtro Savitzky-Golay para redução de ruído.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados
    col : str ou list
        Coluna(s) para aplicar o filtro
    n_window : int
        Tamanho da janela
    degree : int
        Grau do polinômio
        
    Returns:
    --------
    numpy.ndarray
        Valores filtrados
    """
    values = df[col].values
    values = savgol_filter(values, n_window, degree, axis=0)
    return values

def get_features_correlation(df, n_print=50, verbose=True, abs=False):
    """
    Calcula correlações entre features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame com os dados
    n_print : int
        Número de correlações a mostrar
    verbose : bool
        Se deve imprimir informações
    abs : bool
        Se deve usar valores absolutos
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame com correlações
    """
    if abs:
        correlation_matrix = df.corr().abs()
    else:
        correlation_matrix = df.corr()
    
    triangular_mask = np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    correlation_df = correlation_matrix.where(triangular_mask).stack().reset_index()
    
    if abs:
        correlation_df.columns = ['Feature 01', 'Feature 02', 'Correlation [absolute]']
        correlation_df.sort_values(by='Correlation [absolute]', ascending=False, inplace=True)
    else:
        correlation_df.columns = ['Feature 01', 'Feature 02', 'Correlation']
        correlation_df.sort_values(by='Correlation', ascending=False, inplace=True)
    
    if verbose:
        print('>'*10, f' FEATURES COLLINEARITY (Showing TOP {n_print})\n', correlation_df.head(n_print), '\n')
    
    return correlation_df

def plot_confusion_matrix(y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
    """
    Plota a matriz de confusão.
    
    Parameters:
    -----------
    y_true : array-like
        Valores verdadeiros
    y_pred : array-like
        Valores preditos
    classes : list
        Lista de nomes das classes
    normalize : bool
        Se deve normalizar os valores
    title : str
        Título do gráfico
    cmap : matplotlib.colors.Colormap
        Mapa de cores para o gráfico
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

def plot_roc_curve(y_true, y_score, title='Receiver Operating Characteristic (ROC) Curve'):
    """
    Plota a curva ROC.
    
    Parameters:
    -----------
    y_true : array-like
        Valores verdadeiros
    y_score : array-like
        Scores preditos (probabilidades)
    title : str
        Título do gráfico
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()

def print_classification_report(y_true, y_pred, target_names=None):
    """
    Imprime o relatório de classificação.
    
    Parameters:
    -----------
    y_true : array-like
        Valores verdadeiros
    y_pred : array-like
        Valores preditos
    target_names : list, opcional
        Lista com os nomes das classes
    """
    print(classification_report(y_true, y_pred, target_names=target_names))
