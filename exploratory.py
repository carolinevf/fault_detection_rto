"""
Classe para análise exploratória de dados.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
import missingno as msno
from rto_analysis.config import FIGURES_DIR

logger = logging.getLogger(__name__)

class ExploratoryAnalysis:
    def __init__(self, df):
        """
        Inicializa o analisador com um DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame a ser analisado
        """
        self.df = df
        logger.info(f"Análise exploratória inicializada com DataFrame de shape {df.shape}")
    
    def plot_time_series(self, column, figsize=(12, 3), save_path=None):
        """
        Plota série temporal de uma coluna.
        
        Parameters:
        -----------
        column : str
            Coluna para plotar
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        plt.figure(figsize=figsize)
        sns.lineplot(data=self.df, x=self.df.index, y=column)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"timeseries_{column}.png")
            logger.info(f"Gráfico de série temporal salvo em {save_path}/timeseries_{column}.png")
        plt.close()
    
    def plot_distribution(self, column, figsize=(12, 3), save_path=None):
        """
        Plota distribuição de uma coluna.
        
        Parameters:
        -----------
        column : str
            Coluna para plotar
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        plt.figure(figsize=figsize)
        sns.histplot(data=self.df, x=column)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"dist_{column}.png")
            logger.info(f"Gráfico de distribuição salvo em {save_path}/dist_{column}.png")
        plt.close()
    
    def plot_combined_analysis(self, column, target_col='ST', figsize=(12, 6), save_path=None):
        """
        Plota análise combinada de uma coluna.
        
        Parameters:
        -----------
        column : str
            Coluna para plotar
        target_col : str
            Coluna alvo
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Boxplot por target
        sns.boxplot(data=self.df, x=target_col, y=column, showfliers=False, ax=axes[0, 0])
        
        # Série temporal
        sns.lineplot(data=self.df, x=self.df.index, y=column, ax=axes[0, 1])
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Distribuição por target
        sns.kdeplot(data=self.df, x=column, hue=target_col, ax=axes[1, 0])
        
        # ECDF por target
        sns.ecdfplot(data=self.df, x=column, hue=target_col, stat='proportion', ax=axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"combined_{column}.png")
            logger.info(f"Gráfico de análise combinada salvo em {save_path}/combined_{column}.png")
        plt.close()
    
    def plot_missing_values_matrix(self, figsize=(12, 8), save_path=None):
        """
        Plota matriz de valores ausentes.
        
        Parameters:
        -----------
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        plt.figure(figsize=figsize)
        msno.matrix(self.df, sparkline=True, fontsize=10)
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / "missing_matrix.png")
            logger.info(f"Matriz de valores ausentes salva em {save_path}/missing_matrix.png")
        plt.close()
    
    def plot_missing_values_heatmap(self, figsize=(12, 10), save_path=None):
        """
        Plota heatmap de correlação de valores ausentes.
        
        Parameters:
        -----------
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        plt.figure(figsize=figsize)
        msno.heatmap(self.df, fontsize=12)
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / "missing_heatmap.png")
            logger.info(f"Heatmap de correlação de valores ausentes salvo em {save_path}/missing_heatmap.png")
        plt.close()
    
    def plot_correlation_heatmap(self, figsize=(15, 10), save_path=None):
        """
        Plota heatmap de correlação entre variáveis.
        
        Parameters:
        -----------
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        plt.figure(figsize=figsize)
        corr = self.df.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, cmap='BrBG', annot=False, center=0)
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / "correlation_heatmap.png")
            logger.info(f"Heatmap de correlação salvo em {save_path}/correlation_heatmap.png")
        plt.close()
    
    def analyze_by_equipment(self, tag_mapping, save_path=None):
        """
        Analisa variáveis agrupadas por equipamento.
        
        Parameters:
        -----------
        tag_mapping : dict
            Dicionário mapeando equipamentos para suas tags
        save_path : str, opcional
            Caminho para salvar as figuras
        """
        for equipment, tags in tag_mapping.items():
            logger.info(f"Analisando equipamento: {equipment}")
            
            # Verifica quais tags estão presentes no DataFrame
            valid_tags = [tag for tag in tags if tag in self.df.columns]
            
            if not valid_tags:
                logger.warning(f"Nenhuma tag válida encontrada para o equipamento {equipment}")
                continue
            
            # Cria pasta específica para o equipamento
            if save_path:
                equipment_path = Path(save_path) / equipment
                equipment_path.mkdir(parents=True, exist_ok=True)
            else:
                equipment_path = None
            
            # Análise para cada tag
            for tag in valid_tags:
                self.plot_combined_analysis(tag, save_path=equipment_path)
    
    def analyze_by_quartiles(self, column, target_col='ST', figsize=(12, 3), save_path=None):
        """
        Analisa uma variável por quartis em relação ao target.
        
        Parameters:
        -----------
        column : str
            Coluna para analisar
        target_col : str
            Coluna alvo
        figsize : tuple
            Tamanho da figura
        save_path : str, opcional
            Caminho para salvar a figura
        """
        temp_df = self.df.copy()
        
        # Criar categorias por quartis
        cat_col = f'cat_{column}'
        temp_df[cat_col] = pd.qcut(temp_df[column], q=[0, .25, .5, .75, 1.]).sort_values()
        temp_df[cat_col] = temp_df[cat_col].apply(lambda x: pd.Interval(left=round(x.left, 1), right=round(x.right, 1)))
        
        # Plotar
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        sns.kdeplot(data=temp_df, x=column, hue=target_col, ax=axes[0])
        sns.boxplot(data=temp_df, x=cat_col, y=column, hue=target_col, ax=axes[1])
        axes[1].tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path / f"quartile_{column}.png")
            logger.info(f"Análise por quartis salva em {save_path}/quartile_{column}.png")
        plt.close()
        
        return temp_df[[column, cat_col, target_col]]
