#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para carregar e usar um modelo LSTM salvo em formato .h5
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

def carregar_modelo(caminho_modelo='melhor_modelo_lstm.h5'):
    """
    Carrega o modelo LSTM salvo
    
    Args:
        caminho_modelo: Caminho para o arquivo .h5 do modelo
    
    Returns:
        model: Modelo carregado
    """
    try:
        print(f"Carregando modelo de: {caminho_modelo}")
        model = load_model(caminho_modelo)
        print("✓ Modelo carregado com sucesso!")
        print(f"Arquitetura do modelo:")
        model.summary()
        return model
    except Exception as e:
        print(f"✗ Erro ao carregar o modelo: {e}")
        return None

def carregar_scaler(caminho_scaler='scaler.pkl'):
    """
    Carrega o scaler salvo (se existir)
    
    Args:
        caminho_scaler: Caminho para o arquivo do scaler
    
    Returns:
        scaler: Scaler carregado ou None
    """
    try:
        import pickle
        with open(caminho_scaler, 'rb') as f:
            scaler = pickle.load(f)
        print("✓ Scaler carregado com sucesso!")
        return scaler
    except:
        print("⚠️  Scaler não encontrado. Será criado um novo.")
        return None

def preparar_dados_para_predicao(dados, sequence_length=10, scaler=None):
    """
    Prepara dados para predição
    
    Args:
        dados: DataFrame ou array com os dados
        sequence_length: Comprimento da sequência
        scaler: Scaler para normalização
    
    Returns:
        X_sequences: Dados preparados para predição
    """
    # Se dados for DataFrame, extrai as features
    if isinstance(dados, pd.DataFrame):
        features = ['look_vec_x', 'look_vec_y', 'look_vec_z', 'pupil_size']
        dados_array = dados[features].values
    else:
        dados_array = dados
    
    # Remove valores nulos
    dados_clean = dados_array[~np.isnan(dados_array).any(axis=1)]
    
    # Normalização
    if scaler is None:
        scaler = StandardScaler()
        dados_scaled = scaler.fit_transform(dados_clean)
    else:
        dados_scaled = scaler.transform(dados_clean)
    
    # Criação de sequências
    X_sequences = []
    for i in range(len(dados_scaled) - sequence_length + 1):
        X_sequences.append(dados_scaled[i:i + sequence_length])
    
    return np.array(X_sequences), scaler

def fazer_predicao(model, dados, sequence_length=10, scaler=None):
    """
    Faz predições usando o modelo carregado
    
    Args:
        model: Modelo carregado
        dados: Dados para predição
        sequence_length: Comprimento da sequência
        scaler: Scaler para normalização
    
    Returns:
        predictions: Predições do modelo
        probabilities: Probabilidades das predições
    """
    # Prepara os dados
    X_sequences, scaler = preparar_dados_para_predicao(dados, sequence_length, scaler)
    
    if len(X_sequences) == 0:
        print("✗ Não há dados suficientes para fazer predições")
        return None, None
    
    print(f"Fazendo predições para {len(X_sequences)} sequências...")
    
    # Faz as predições
    probabilities = model.predict(X_sequences)
    predictions = np.argmax(probabilities, axis=1)
    
    return predictions, probabilities

def interpretar_predicoes(predictions, probabilities):
    """
    Interpreta e exibe as predições
    
    Args:
        predictions: Predições do modelo
        probabilities: Probabilidades das predições
    """
    if predictions is None:
        return
    
    print(f"\n=== RESULTADOS DAS PREDIÇÕES ===")
    print(f"Total de predições: {len(predictions)}")
    
    # Conta as predições
    unique, counts = np.unique(predictions, return_counts=True)
    pred_dict = dict(zip(unique, counts))
    
    print(f"\nDistribuição das predições:")
    print(f"- Sem Atenção (0): {pred_dict.get(0, 0)} ({pred_dict.get(0, 0)/len(predictions)*100:.1f}%)")
    print(f"- Com Atenção (1): {pred_dict.get(1, 0)} ({pred_dict.get(1, 0)/len(predictions)*100:.1f}%)")
    
    # Mostra algumas predições detalhadas
    print(f"\nPrimeiras 10 predições detalhadas:")
    for i in range(min(10, len(predictions))):
        prob_sem_atencao = probabilities[i][0]
        prob_com_atencao = probabilities[i][1]
        pred = "Com Atenção" if predictions[i] == 1 else "Sem Atenção"
        confianca = max(prob_sem_atencao, prob_com_atencao)
        
        print(f"  Sequência {i+1}: {pred} (confiança: {confianca:.3f})")

def carregar_dados_exemplo():
    """
    Carrega dados de exemplo do arquivo CSV
    """
    try:
        print("Carregando dados de exemplo...")
        df = pd.read_csv('gaze_labels.csv')
        df.columns = df.columns.str.strip()
        print(f"✓ Dados carregados: {len(df)} registros")
        return df
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return None

def main():
    """
    Função principal
    """
    print("=== CARREGAMENTO E USO DE MODELO LSTM ===")
    
    # 1. Carrega o modelo
    model = carregar_modelo()
    if model is None:
        print("Não foi possível carregar o modelo. Verifique se o arquivo existe.")
        return
    
    # 2. Carrega o scaler (se existir)
    scaler = carregar_scaler()
    
    # 3. Carrega dados de exemplo
    dados = carregar_dados_exemplo()
    if dados is None:
        print("Não foi possível carregar dados de exemplo.")
        return
    
    # 4. Faz predições
    predictions, probabilities = fazer_predicao(model, dados, sequence_length=10, scaler=scaler)
    
    # 5. Interpreta os resultados
    interpretar_predicoes(predictions, probabilities)
    
    print(f"\n=== RESUMO ===")
    print(f"Modelo carregado com sucesso!")
    print(f"Predições realizadas para {len(predictions) if predictions is not None else 0} sequências")

def exemplo_uso_individual():
    """
    Exemplo de como usar o modelo para predições individuais
    """
    print("\n=== EXEMPLO DE USO INDIVIDUAL ===")
    
    # Carrega o modelo
    model = carregar_modelo()
    if model is None:
        return
    
    # Dados de exemplo (4 features: look_vec_x, look_vec_y, look_vec_z, pupil_size)
    dados_exemplo = np.array([
        [0.1, 0.2, -0.9, 0.05],  # Sequência 1
        [0.2, 0.1, -0.8, 0.06],  # Sequência 2
        [0.0, 0.0, -1.0, 0.04],  # Sequência 3
        [0.1, 0.1, -0.95, 0.05], # Sequência 4
        [0.0, 0.0, -1.0, 0.04],  # Sequência 5
        [0.1, 0.2, -0.9, 0.05],  # Sequência 6
        [0.2, 0.1, -0.8, 0.06],  # Sequência 7
        [0.0, 0.0, -1.0, 0.04],  # Sequência 8
        [0.1, 0.1, -0.95, 0.05], # Sequência 9
        [0.0, 0.0, -1.0, 0.04]   # Sequência 10
    ])
    
    print("Dados de exemplo:")
    print(dados_exemplo)
    
    # Faz predição
    predictions, probabilities = fazer_predicao(model, dados_exemplo, sequence_length=10)
    
    if predictions is not None:
        print(f"\nPredição para dados de exemplo:")
        pred = "Com Atenção" if predictions[0] == 1 else "Sem Atenção"
        confianca = max(probabilities[0])
        print(f"Resultado: {pred} (confiança: {confianca:.3f})")

if __name__ == "__main__":
    # Executa o exemplo principal
    main()
    
    # Executa o exemplo individual
    exemplo_uso_individual() 