import os
import time
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import tensorflow as tf

# --- Configurações ---
# Caminho para o modelo treinado (ajuste para o modelo que você quer usar)
MODEL_PATH = './models/gaze_attention_model_finetuned_regression.keras'
IMG_SIZE = (64, 64) # O mesmo tamanho usado no treinamento do modelo
TIME_WINDOW = 10    # O mesmo TIME_WINDOW usado para criar as sequências

# --- Carregar o modelo ---
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Modelo carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Verifique se o caminho do modelo está correto.")
    exit()

# --- Configurar a Webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

# --- Variáveis para o loop de inferência e logging ---
current_fixation_start = None
fixations_log = []
sequence_frames = []

print("Aplicação iniciada. Pressione 'q' para sair.")
print("Iniciando detecção de atenção...")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar a imagem para o modelo
        frame_resized = cv2.resize(frame, IMG_SIZE)
        frame_normalized = frame_resized.astype('float32') / 255.0

        # Acumular frames para a sequência
        sequence_frames.append(frame_normalized)
        if len(sequence_frames) > TIME_WINDOW:
            sequence_frames.pop(0)

        # Fazer inferência apenas quando a sequência estiver completa
        if len(sequence_frames) == TIME_WINDOW:
            input_sequence = np.expand_dims(np.array(sequence_frames), axis=0)
            prediction = model.predict(input_sequence, verbose=0)[0][0]
            
            current_time = time.time()
            is_attentive = prediction > 0.5

            # --- LOGS DE DEPURAÇÃO ---
            print(f"Previsão: {prediction:.2f} -> Atento: {is_attentive}")
            
            if is_attentive:
                if current_fixation_start is None:
                    current_fixation_start = current_time
                    print(f"DEBUG: Fixacao iniciada em {current_fixation_start}")
                
                status_text = "Atento"
                status_color = (0, 255, 0)
            else:
                if current_fixation_start is not None:
                    fixation_duration = current_time - current_fixation_start
                    fixations_log.append({
                        'timestamp_start': datetime.fromtimestamp(current_fixation_start).strftime('%Y-%m-%d %H:%M:%S'),
                        'duration_seconds': fixation_duration
                    })
                    print(f"DEBUG: Fixacao registrada. Duracao: {fixation_duration:.2f}s")
                
                current_fixation_start = None
                status_text = "Nao Atento"
                status_color = (0, 0, 255)

            cv2.putText(frame, f"Status: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            cv2.putText(frame, f"Probabilidade: {prediction:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        cv2.imshow('Analise de Atencao', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nAplicacao encerrada pelo usuario.")
            break

finally:
    print(f"\nDEBUG: A lista de logs de fixacao contem {len(fixations_log)} registros.")
    
    if current_fixation_start is not None:
        fixation_duration = time.time() - current_fixation_start
        fixations_log.append({
            'timestamp_start': datetime.fromtimestamp(current_fixation_start).strftime('%Y-%m-%d %H:%M:%S'),
            'duration_seconds': fixation_duration
        })
        print(f"DEBUG: Fixacao final registrada. Duracao: {fixation_duration:.2f}s")
        
    if fixations_log:
        log_filename = f"log_atencao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        log_df = pd.DataFrame(fixations_log)
        log_df.to_csv(log_filename, index=False)
        print(f"\nLog de atencao salvo em: {log_filename}")
    
    cap.release()
    cv2.destroyAllWindows()