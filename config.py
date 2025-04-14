"""
Configurações globais para o projeto.
"""

from pathlib import Path

# Diretórios
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
CLEANED_DATA_DIR = DATA_DIR / 'cleaned'
FIGURES_DIR = PROJECT_DIR / 'figures'
MODELS_DIR = PROJECT_DIR / 'models'

# Parâmetros para pré-processamento
KNN_IMPUTER_NEIGHBORS = 11 ## verificar
SAVGOL_WINDOW = 31
SAVGOL_DEGREE = 2
CORRELATION_THRESHOLD = 0.8
N_BEST_FEATURES = 30

# Parâmetros para modelagem
TEST_SIZE = 0.3
RANDOM_STATE = 42
