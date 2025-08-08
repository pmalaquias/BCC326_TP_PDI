#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simples para carregar e usar um modelo LSTM salvo em .h5
"""

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

def carregar_e_usar_modelo():
    """
    Carrega o modelo e faz predições
    """
    # 1. Carrega o modelo
    try:
        print("Carregando modelo...")
        model = load_model('melhor_modelo_lstm.h5')
        print("✓ Modelo carregado com sucesso!")
    except Exception as e:
        print(f"✗ Erro ao carregar modelo: {e}")
        print("Certifique-se de que o arquivo 'melhor_modelo_lstm.h5' existe.")
        return
    
    # 2. Carrega dados de exemplo
    try:
        print("Carregando dados...")
        df = pd.read_csv('gaze_labels.csv')
        df.columns = df.columns.str.strip()
        print(f"✓ Dados carregados: {len(df)} registros")
    except Exception as e:
        print(f"✗ Erro ao carregar dados: {e}")
        return
    
    # 3. Prepara os dados
    features = ['look_vec_x', 'look_vec_y', 'look_vec_z', 'pupil_size']
    dados = df[features].dropna().values
    
    # Normalização
    scaler = StandardScaler()
    dados_scaled = scaler.fit_transform(dados)
    
    # Cria sequências (10 timesteps)
    sequence_length = 10
    X_sequences = []
    
    for i in range(len(dados_scaled) - sequence_length + 1):
        X_sequences.append(dados_scaled[i:i + sequence_length])
    
    X_sequences = np.array(X_sequences)
    print(f"✓ Sequências criadas: {X_sequences.shape}")
    
    # 4. Faz predições
    print("Fazendo predições...")
    predictions = model.predict(X_sequences)
    predicted_classes = np.argmax(predictions, axis=1)
    
    # 5. Mostra resultados
    print(f"\n=== RESULTADOS ===")
    print(f"Total de predições: {len(predicted_classes)}")
    
    # Conta as predições
    sem_atencao = np.sum(predicted_classes == 0)
    com_atencao = np.sum(predicted_classes == 1)
    
    print(f"Sem atenção: {sem_atencao} ({sem_atencao/len(predicted_classes)*100:.1f}%)")
    print(f"Com atenção: {com_atencao} ({com_atencao/len(predicted_classes)*100:.1f}%)")
    
    # Mostra algumas predições
    print(f"\nPrimeiras 5 predições:")
    for i in range(min(5, len(predicted_classes))):
        prob = predictions[i]
        classe = "Com Atenção" if predicted_classes[i] == 1 else "Sem Atenção"
        confianca = max(prob)
        print(f"  Sequência {i+1}: {classe} (confiança: {confianca:.3f})")

def exemplo_predicao_individual():
    """
    Exemplo de como fazer uma predição individual
    """
    print("\n=== EXEMPLO DE PREDIÇÃO INDIVIDUAL ===")
    
    # Carrega o modelo
    try:
        model = load_model('melhor_modelo_lstm.h5')
    except:
        print("Modelo não encontrado. Execute primeiro o treinamento.")
        return
    
    # Dados de exemplo (10 sequências de 4 features cada)
    dados_exemplo = np.array([
        [0.1, 0.2, -0.9, 0.05],  # look_vec_x, look_vec_y, look_vec_z, pupil_size
        [0.2, 0.1, -0.8, 0.06],
        [0.0, 0.0, -1.0, 0.04],
        [0.1, 0.1, -0.95, 0.05],
        [0.0, 0.0, -1.0, 0.04],
        [0.1, 0.2, -0.9, 0.05],
        [0.2, 0.1, -0.8, 0.06],
        [0.0, 0.0, -1.0, 0.04],
        [0.1, 0.1, -0.95, 0.05],
        [0.0, 0.0, -1.0, 0.04]
    ])
    
    # Normaliza os dados
    scaler = StandardScaler()
    dados_scaled = scaler.fit_transform(dados_exemplo)
    
    # Cria sequência (adiciona dimensão de batch)
    X_sequence = dados_scaled.reshape(1, 10, 4)
    
    # Faz predição
    prediction = model.predict(X_sequence)
    predicted_class = np.argmax(prediction[0])
    confidence = max(prediction[0])
    
    resultado = "Com Atenção" if predicted_class == 1 else "Sem Atenção"
    
    print(f"Dados de entrada:")
    print(dados_exemplo)
    print(f"\nPredição: {resultado}")
    print(f"Confiança: {confidence:.3f}")
    print(f"Probabilidades: Sem atenção={prediction[0][0]:.3f}, Com atenção={prediction[0][1]:.3f}")

if __name__ == "__main__":
    # Executa o exemplo principal
    carregar_e_usar_modelo()
    
    # Executa o exemplo individual
    exemplo_predicao_individual() 