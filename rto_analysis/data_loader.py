"""
Funções para carregamento de dados.
"""

import pandas as pd
import logging
from pathlib import Path
from rto_analysis.config import RAW_DATA_DIR

logger = logging.getLogger(__name__)

def load_data(filepath=None):
    """
    Carrega dados do arquivo CSV.
    
    Parameters:
    -----------
    filepath : str, opcional
        Caminho para o arquivo de dados. Se None, usa o padrão.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame contendo os dados carregados
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / 'df_full_26092021.csv'
    else:
        filepath = Path(filepath)
    
    logger.info(f"Carregando dados de {filepath}")
    
    if filepath.suffix == '.csv':
        return pd.read_csv(filepath)
    elif filepath.suffix in ['.xlsx', '.xls']:
        return pd.read_excel(filepath)
    elif filepath.suffix == '.parquet':
        return pd.read_parquet(filepath)
    else:
        raise ValueError(f"Formato de arquivo não suportado: {filepath.suffix}")

def load_tag_descriptions(filepath=None):
    """
    Carrega descrições das tags.
    
    Parameters:
    -----------
    filepath : str, opcional
        Caminho para o arquivo de descrição das tags. Se None, usa o padrão.
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame contendo as descrições das tags
    """
    if filepath is None:
        filepath = RAW_DATA_DIR / 'descricao_tags_essa.xlsx'
    else:
        filepath = Path(filepath)
    
    logger.info(f"Carregando descrições das tags de {filepath}")
    
    return pd.read_excel(filepath, sheet_name='TAGS', skiprows=1)
