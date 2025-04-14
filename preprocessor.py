"""
Classe para pré-processamento de dados.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer
from scipy.signal import savgol_filter
from sklearn.feature_selection import mutual_info_classif
from rto_analysis.utils import get_features_correlation, apply_savgol_filter

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, df):
        """
        Inicializa o preprocessador com um DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame a ser processado
        """
        self.df_original = df.copy()
        self.df = df.copy()
        self.removed_cols = []
        self.scaler = None
        self.imputer = None
        logger.info(f"Preprocessador inicializado com DataFrame de shape {df.shape}")
    
    def remove_controlled_variables(self, pattern='C'):
        """
        Remove variáveis controladas com base em um padrão.
        
        Parameters:
        -----------
        pattern : str
            Padrão para identificar variáveis controladas
            
        Returns:
        --------
        self
        """
        controlled_vars = self.df.columns[self.df.columns.str.contains(pat=pattern)].tolist()
        self.removed_cols.extend(controlled_vars)
        self.df = self.df.drop(controlled_vars, axis=1)
        logger.info(f"Removidas {len(controlled_vars)} variáveis controladas")
        return self
    
    def remove_empty_columns(self):
        """
        Remove colunas com 100% de valores ausentes.
        
        Returns:
        --------
        self
        """
        empty_cols = self.df.columns[self.df.isnull().mean() == 1].tolist()
        self.removed_cols.extend(empty_cols)
        self.df = self.df.drop(empty_cols, axis=1)
        logger.info(f"Removidas {len(empty_cols)} colunas vazias")
        return self
    
    def remove_specific_columns(self, columns_to_remove):
        """
        Remove colunas específicas.
        
        Parameters:
        -----------
        columns_to_remove : list
            Lista de colunas a remover
            
        Returns:
        --------
        self
        """
        valid_cols = [col for col in columns_to_remove if col in self.df.columns]
        self.removed_cols.extend(valid_cols)
        self.df = self.df.drop(valid_cols, axis=1)
        logger.info(f"Removidas {len(valid_cols)} colunas específicas")
        return self
    
    def impute_missing_values(self, method='knn', n_neighbors=11, weights='distance'):
        """
        Imputa valores ausentes.
        
        Parameters:
        -----------
        method : str
            Método de imputação ('knn', 'mean', 'median', 'ffill', 'bfill')
        n_neighbors : int
            Número de vizinhos para KNN
        weights : str
            Pesos para KNN ('uniform', 'distance')
            
        Returns:
        --------
        self
        """
        logger.info(f"Imputando valores ausentes usando método {method}")
        
        if method == 'knn':
            # Normalizar dados para KNN
            self.scaler = MinMaxScaler()
            df_scaled = pd.DataFrame(
                self.scaler.fit_transform(self.df), 
                columns=self.df.columns,
                index=self.df.index
            )
            
            # Aplicar KNN imputer
            self.imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
            df_imputed_scaled = pd.DataFrame(
                self.imputer.fit_transform(df_scaled),
                columns=df_scaled.columns,
                index=df_scaled.index
            )
            
            # Reverter normalização
            self.df = pd.DataFrame(
                self.scaler.inverse_transform(df_imputed_scaled),
                columns=df_imputed_scaled.columns,
                index=df_imputed_scaled.index
            )
        elif method == 'mean':
            self.df = self.df.fillna(self.df.mean())
        elif method == 'median':
            self.df = self.df.fillna(self.df.median())
        elif method == 'ffill':
            self.df = self.df.fillna(method='ffill')
        elif method == 'bfill':
            self.df = self.df.fillna(method='bfill')
        else:
            raise ValueError("Método de imputação não suportado")
        
        return self
    
    def apply_noise_filter(self, columns=None, n_window=31, degree=2):
        """
        Aplica filtro para redução de ruído.
        
        Parameters:
        -----------
        columns : list, opcional
            Colunas para aplicar o filtro (None = todas)
        n_window : int
            Tamanho da janela
        degree : int
            Grau do polinômio
            
        Returns:
        --------
        self
        """
        logger.info(f"Aplicando filtro de ruído com janela {n_window} e grau {degree}")
        
        if columns is None:
            columns = self.df.columns.tolist()
            if 'ST' in columns:
                columns.remove('ST')  # Não aplicar filtro na coluna alvo
        
        filtered_data = apply_savgol_filter(self.df, columns, n_window, degree)
        self.df[columns] = filtered_data
        
        return self
    
    def remove_highly_correlated_features(self, threshold=0.8):
        """
        Remove features altamente correlacionadas.
        
        Parameters:
        -----------
        threshold : float
            Limiar de correlação (0.0-1.0)
            
        Returns:
        --------
        self
        """
        logger.info(f"Removendo features com correlação > {threshold}")
        
        corr_df = get_features_correlation(self.df, verbose=False, abs=True)
        high_corr_features = corr_df[
            corr_df['Correlation [absolute]'] > threshold
        ]['Feature 02'].drop_duplicates().tolist()
        
        self.removed_cols.extend(high_corr_features)
        self.df = self.df.drop(high_corr_features, axis=1)
        
        logger.info(f"Removedas {len(high_corr_features)} features altamente correlacionadas")
