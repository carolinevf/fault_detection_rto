#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script principal para execução da análise de dados RTO.
Este script coordena todas as etapas do processo de análise:
1. Carregamento dos dados
2. Pré-processamento
3. Análise exploratória
4. Modelagem
5. Avaliação
"""

import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Importando módulos do nosso pacote
from rto_analysis.data_loader import load_data, load_tag_descriptions
from rto_analysis.preprocessor import DataPreprocessor
from rto_analysis.exploratory import ExploratoryAnalysis
from rto_analysis.modeling import train_models, evaluate_models
from rto_analysis.utils import setup_logging, apply_savgol_filter, get_features_correlation

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Análise de dados RTO')
    parser.add_argument('--skip-preprocessing', action='store_true', help='Pular etapa de pré-processamento')
    parser.add_argument('--skip-exploration', action='store_true', help='Pular etapa de análise exploratória')
    parser.add_argument('--skip-modeling', action='store_true', help='Pular etapa de modelagem')
    parser.add_argument('--data-path', type=str, default='data/raw/df_full_26092021.csv', help='Caminho para o arquivo de dados')
    parser.add_argument('--tags-path', type=str, default='data/raw/descricao_tags_essa.xlsx', help='Caminho para o arquivo de descrição das tags')
    parser.add_argument('--output-dir', type=str, default='data/processed', help='Diretório para salvar os dados processados')
    return parser.parse_args()

def main():
    """Função principal que executa todas as etapas do pipeline."""
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Iniciar cronômetro
    start_time = time.time()
    logger.info("Iniciando análise de dados RTO")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Etapa 1: Carregamento dos dados
    logger.info("Etapa 1: Carregamento dos dados")
    df = load_data(args.data_path)
    tags_description = load_tag_descriptions(args.tags_path)
    logger.info(f"Dados carregados com sucesso: {df.shape[0]} linhas e {df.shape[1]} colunas")
    
    # Etapa 2: Pré-processamento
    if not args.skip_preprocessing:
        logger.info("Etapa 2: Pré-processamento dos dados")
        preprocessor = DataPreprocessor(df)
        
        # Remoção de variáveis controladas
        preprocessor.remove_controlled_variables(pattern='C')
        logger.info(f"Variáveis controladas removidas. Restantes: {preprocessor.df.shape[1]} colunas")
        
        # Remoção de colunas vazias
        preprocessor.remove_empty_columns()
        logger.info(f"Colunas vazias removidas. Restantes: {preprocessor.df.shape[1]} colunas")
        
        # Remoção de colunas específicas (com comportamentos estranhos)
        drop_negative_var = [
            'PDI445B', 'PDI441B', 'PDI444B', 'PI647', 'PI425B', 'PI645',
            # ... adicione todas as outras colunas a serem removidas
        ]
        preprocessor.remove_specific_columns(drop_negative_var)
        logger.info(f"Colunas específicas removidas. Restantes: {preprocessor.df.shape[1]} colunas")
        
        # Imputação de valores ausentes
        preprocessor.impute_missing_values(method='knn', n_neighbors=11)
        logger.info("Valores ausentes imputados com KNN")
        
        # Aplicação de filtro para redução de ruído
        preprocessor.apply_noise_filter(n_window=31, degree=2)
        logger.info("Filtro Savitzky-Golay aplicado para redução de ruído")
        
        # Remoção de features altamente correlacionadas
        preprocessor.remove_highly_correlated_features(threshold=0.8)
        logger.info(f"Features altamente correlacionadas removidas. Restantes: {preprocessor.df.shape[1]} colunas")
        
        # Seleção das melhores features
        preprocessor.select_best_features(target_col='ST', n_features=30)
        logger.info(f"Seleção das melhores features concluída. Features selecionadas: {preprocessor.df.shape[1]-1}")
        
        # Obter dados processados
        df_processed = preprocessor.get_processed_data()
        
        # Salvar dados processados
        output_path = Path(args.output_dir) / 'processed_data.parquet'
        df_processed.to_parquet(output_path)
        logger.info(f"Dados processados salvos em {output_path}")
    else:
        # Carregar dados pré-processados se a etapa for pulada
        output_path = Path(args.output_dir) / 'processed_data.parquet'
        df_processed = pd.read_parquet(output_path)
        logger.info(f"Carregando dados pré-processados de {output_path}")
    
    # Etapa 3: Análise Exploratóri
    if not args.skip_exploration:
        logger.info("Etapa 3: Análise Exploratória de Dados")
        explorer = ExploratoryAnalysis(df_processed)
        
        # Análise de distribuição de variáveis
        logger.info("Gerando gráficos de distribuição")
        for column in df_processed.columns:
            if column != 'ST':  # Excluir a coluna alvo
                explorer.plot_distribution(column, save_path='figures/distributions')
        
        # Análise de séries temporais
        logger.info("Gerando gráficos de séries temporais")
        for column in df_processed.columns:
            if column != 'ST':  # Excluir a coluna alvo
                explorer.plot_time_series(column, save_path='figures/timeseries')
        
        # Análise combinada
        logger.info("Gerando análises combinadas")
        for column in df_processed.columns:
            if column != 'ST':  # Excluir a coluna alvo
                explorer.plot_combined_analysis(column, save_path='figures/combined')
        
        # Matriz de correlação
        logger.info("Gerando matriz de correlação")
        explorer.plot_correlation_heatmap(save_path='figures/correlation')
        
        logger.info("Análise exploratória concluída")
    
    # Etapa 4: Modelagem
    if not args.skip_modeling:
        logger.info("Etapa 4: Modelagem")
        
        # Dividir dados em treino e teste
        X = df_processed.drop('ST', axis=1)
        y = df_processed['ST']
        
        # Treinar modelos
        models = train_models(X, y)
        logger.info("Modelos treinados com sucesso")
        
        # Avaliar modelos
        results = evaluate_models(models, X, y)
        logger.info("Avaliação dos modelos concluída")
        
        # Exibir resultados
        for model_name, metrics in results.items():
            logger.info(f"Modelo: {model_name}")
            for metric_name, value in metrics.items():
                logger.info(f"  {metric_name}: {value}")
    
    # Finalizar
    elapsed_time = time.time() - start_time
    logger.info(f"Análise concluída em {elapsed_time:.2f} segundos")

if __name__ == "__main__":
    main()
