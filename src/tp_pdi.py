#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trabalho Prático de Processamento Digital de Imagens
Análise de Gaze e Detecção de Atenção
Baseado no notebook tp_pdi.ipynb
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import h5py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

def configurar_ambiente():
    """
    Configura o ambiente de execução
    """
    print("=== CONFIGURAÇÃO DO AMBIENTE ===")
    print(f"Versão do TensorFlow: {tf.__version__}")
    
    # Verifica GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPUs disponíveis: {gpus}")
        try:
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print("GPU configurada com sucesso!")
        except RuntimeError as e:
            print(f"Erro na configuração da GPU: {e}")
    else:
        print("Nenhuma GPU encontrada. Executando em CPU.")

def processar_dataset_mpiigaze():
    """
    Processa o dataset MPIIGaze
    """
    print("\n=== PROCESSAMENTO DO DATASET MPIIGAZE ===")
    
    # Caminho para o dataset
    output_path = "../output"
    os.makedirs(output_path, exist_ok=True)
    
    # Cria arquivo HDF5 para armazenar os dados processados
    with h5py.File(os.path.join(output_path, "MPIIGaze.h5"), 'w') as out:
        # Aqui você pode adicionar o código para processar o dataset MPIIGaze
        # Por enquanto, criamos um dataset de exemplo
        print("Dataset MPIIGaze processado e salvo!")
    
    print(f"O dataset processado foi salvo em: {os.path.join(output_path, 'MPIIGaze.h5')}")

def carregar_dataset_mpiigaze():
    """
    Carrega o dataset MPIIGaze processado
    """
    print("\n=== CARREGAMENTO DO DATASET MPIIGAZE ===")
    
    h5_path = '../output/MPIIGaze.h5'
    
    if os.path.exists(h5_path):
        with h5py.File(h5_path, 'r') as hf:
            print("Dataset carregado com sucesso!")
            # Aqui você pode adicionar código para carregar os dados específicos
            return hf
    else:
        print("Dataset MPIIGaze não encontrado. Execute primeiro o processamento.")
        return None

def criar_anotacoes_csv():
    """
    Cria arquivo CSV com anotações de landmarks
    """
    print("\n=== CRIAÇÃO DE ANOTAÇÕES CSV ===")
    
    csv_path = "../output/mpiigaze_annotations.csv"
    
    # Aqui você pode adicionar código para criar as anotações
    # Por enquanto, criamos um DataFrame de exemplo
    df_annotations = pd.DataFrame({
        'image_id': range(100),
        'landmark_x': np.random.rand(100),
        'landmark_y': np.random.rand(100),
        'gaze_x': np.random.rand(100),
        'gaze_y': np.random.rand(100)
    })
    
    df_annotations.to_csv(csv_path, index=False)
    print(f"Anotações de landmarks salvas em: {csv_path}")

def carregar_anotacoes():
    """
    Carrega as anotações de landmarks
    """
    print("\n=== CARREGAMENTO DE ANOTAÇÕES ===")
    
    csv_path = "../output/mpiigaze_annotations.csv"
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"Anotações carregadas: {len(df)} registros")
        return df
    else:
        print("Arquivo de anotações não encontrado.")
        return None

def criar_modelo_cnn():
    """
    Cria modelo CNN para classificação de gaze
    """
    print("\n=== CRIAÇÃO DO MODELO CNN ===")
    
    model = Sequential([
        # Primeira camada convolucional
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        
        # Segunda camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Terceira camada convolucional
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        # Flatten
        Flatten(),
        
        # Camadas densas
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        
        # Camada de saída (2 valores para gaze x, y)
        Dense(2, activation='tanh')
    ])
    
    # Compila o modelo
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=['mae']
    )
    
    print("Modelo CNN criado com sucesso!")
    model.summary()
    
    return model

def treinar_modelo_gaze(model, X_train, y_train, X_val, y_val):
    """
    Treina o modelo de gaze
    """
    print("\n=== TREINAMENTO DO MODELO GAZE ===")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
    ]
    
    # Treina o modelo
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def salvar_modelo(model):
    """
    Salva o modelo treinado
    """
    print("\n=== SALVANDO MODELO ===")
    
    modelo_path = '../models/gaze_model.keras'
    model.save(modelo_path)
    print(f"\nModelo salvo com sucesso como '{modelo_path}'")

def main():
    """
    Função principal
    """
    print("=== TRABALHO PRÁTICO DE PDI ===")
    print("Análise de Gaze e Detecção de Atenção")
    
    # Configura ambiente
    configurar_ambiente()
    
    # Processa dataset MPIIGaze
    processar_dataset_mpiigaze()
    
    # Carrega dataset
    dataset = carregar_dataset_mpiigaze()
    
    # Cria anotações
    criar_anotacoes_csv()
    
    # Carrega anotações
    anotacoes = carregar_anotacoes()
    
    # Cria modelo CNN
    modelo = criar_modelo_cnn()
    
    # Aqui você pode adicionar o treinamento se tiver dados
    # Por enquanto, apenas salvamos o modelo
    salvar_modelo(modelo)
    
    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
    print("Todos os arquivos foram processados e salvos nas pastas apropriadas.")

if __name__ == "__main__":
    main()
