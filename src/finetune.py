#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning do modelo de atenção usando dados reais do MPIIGaze
Baseado no notebook finetune.ipynb
"""

import os
import numpy as np
import cv2
import scipy.io as sio
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import pandas as pd

def verificar_estrutura_mpiigaze():
    """
    Verifica a estrutura do dataset MPIIGaze
    """
    CAMINHO_BASE = '..'
    MPIIGAZE_DIR = os.path.join(CAMINHO_BASE, 'mpiigaze_real', 'MPIIGaze')
    REAL_DATA_ROOT = os.path.join(MPIIGAZE_DIR, 'Data', 'Normalized')

    try:
        # Pegar o primeiro usuário e o primeiro dia
        user_to_check = sorted([d for d in os.listdir(REAL_DATA_ROOT) if d.startswith('p')])[0]
        day_to_check = sorted([d for d in os.listdir(os.path.join(REAL_DATA_ROOT, user_to_check)) if os.path.isdir(os.path.join(REAL_DATA_ROOT, user_to_check, d))])[0]
        
        # Caminho completo para a pasta de dia
        day_dir_to_check = os.path.join(REAL_DATA_ROOT, user_to_check, day_to_check)

        print(f"Listando o conteúdo da pasta de dia: {day_dir_to_check}")
        
        # Listar o conteúdo do diretório de dia
        for item in os.listdir(day_dir_to_check):
            print(f"  - {item}")

    except (IndexError, FileNotFoundError) as e:
        print("Erro: Não foi possível inspecionar a pasta. Verifique se o dataset foi baixado e descompactado corretamente.")
        print(f"Detalhes do erro: {e}")

def configurar_caminhos():
    """
    Configura os caminhos para os arquivos e diretórios
    """
    # --- Configurações de Caminho (Definitivas) ---
    CAMINHO_BASE = '..'
    MODELO_SALVO = os.path.join(CAMINHO_BASE, 'models', 'gaze_attention_model.keras')
    LABELS_FILE_SYNTHETIC = os.path.join(CAMINHO_BASE, 'output', 'gaze_labels.csv')

    # --- Caminho Corrigido para os Dados Reais do MPIIGaze ---
    MPIIGAZE_DIR = os.path.join(CAMINHO_BASE, 'mpiigaze_real', 'MPIIGaze')
    REAL_DATA_ROOT = os.path.join(MPIIGAZE_DIR, 'Data', 'Normalized')

    # Tamanho da imagem redimensionada
    IMG_SIZE = (64, 64)
    
    return CAMINHO_BASE, MODELO_SALVO, LABELS_FILE_SYNTHETIC, MPIIGAZE_DIR, REAL_DATA_ROOT, IMG_SIZE

def processar_dados_mpiigaze(MPIIGAZE_DIR, REAL_DATA_ROOT, IMG_SIZE):
    """
    Processa os dados reais do MPIIGaze
    """
    print("Verificando se o dataset MPIIGaze já existe...")
    if not os.path.exists(MPIIGAZE_DIR):
        print("Dataset MPIIGaze não encontrado. Por favor, baixe e descompacte o arquivo na raiz do seu projeto.")
        return None, None

    try:
        import scipy.io as sio
    except ImportError:
        print("A biblioteca scipy não está instalada. Instalando...")
        os.system("pip install scipy")
        import scipy.io as sio

    print("Processando dados do MPIIGaze...")
    X_real = []
    y_real = []

    if os.path.exists(REAL_DATA_ROOT):
        USER_NAMES = sorted([d for d in os.listdir(REAL_DATA_ROOT) if d.startswith('p')])
        for user in USER_NAMES:
            user_dir = os.path.join(REAL_DATA_ROOT, user)
            day_folders = sorted([d for d in os.listdir(user_dir) if os.path.isdir(os.path.join(user_dir, d))])
            
            for day in day_folders:
                day_dir = os.path.join(user_dir, day)
                
                annotation_file = os.path.join(day_dir, 'annotation.mat')
                images_folder = os.path.join(day_dir, 'image') # Caminho correto para as imagens

                if os.path.exists(annotation_file) and os.path.exists(images_folder):
                    mat_data = sio.loadmat(annotation_file)
                    gaze_labels = mat_data['gaze']
                    gaze_threshold = 0.1
                    labels = np.linalg.norm(gaze_labels, axis=1) < gaze_threshold
                    
                    # Nomes de arquivos de imagem no formato 0.jpg, 1.jpg, etc.
                    image_filenames = sorted([f for f in os.listdir(images_folder) if f.endswith('.jpg')])
                    
                    # Certifique-se de que o número de imagens e rótulos corresponde
                    if len(image_filenames) == len(labels):
                        for i, image_filename in enumerate(image_filenames):
                            image_path = os.path.join(images_folder, image_filename)
                            img = cv2.imread(image_path)
                            if img is not None:
                                img_resized = cv2.resize(img, IMG_SIZE)
                                img_normalized = img_resized.astype('float32') / 255.0
                                X_real.append(img_normalized)
                                y_real.append(labels[i])

    X_real = np.array(X_real)
    y_real = np.array(y_real)
    print(f"Dados reais processados. Total de imagens: {len(X_real)}")

    if len(X_real) == 0:
        print("Nenhuma imagem foi processada. Verifique se o nome das subpastas é 'image' e se os nomes dos arquivos são '0.jpg', '1.jpg', etc.")
        return None, None

    # O MPIIGaze é um dataset grande, vamos usar uma pequena parte para o ajuste fino.
    X_real_train, X_real_test, y_real_train, y_real_test = train_test_split(X_real, y_real, test_size=0.2, random_state=42)
    
    return (X_real_train, X_real_test, y_real_train, y_real_test)

def verificar_gpu():
    """
    Verifica a disponibilidade de GPU
    """
    import tensorflow as tf

    # Lista todos os dispositivos disponíveis (CPU e GPU)
    print(tf.config.list_physical_devices('GPU'))

    # Confirma se o TensorFlow está compilado com suporte a GPU
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def main():
    """
    Função principal
    """
    print("=== FINE-TUNING DO MODELO DE ATENÇÃO ===")
    
    # Verifica estrutura do MPIIGaze
    verificar_estrutura_mpiigaze()
    
    # Configura caminhos
    CAMINHO_BASE, MODELO_SALVO, LABELS_FILE_SYNTHETIC, MPIIGAZE_DIR, REAL_DATA_ROOT, IMG_SIZE = configurar_caminhos()
    
    # Processa dados do MPIIGaze
    dados_mpiigaze = processar_dados_mpiigaze(MPIIGAZE_DIR, REAL_DATA_ROOT, IMG_SIZE)
    
    # Verifica GPU
    verificar_gpu()
    
    if dados_mpiigaze is not None:
        X_real_train, X_real_test, y_real_train, y_real_test = dados_mpiigaze
        print(f"Dados de treino: {X_real_train.shape}")
        print(f"Dados de teste: {X_real_test.shape}")
        print("Fine-tuning pode ser executado com sucesso!")
    else:
        print("Não foi possível processar os dados do MPIIGaze.")

if __name__ == "__main__":
    main()
