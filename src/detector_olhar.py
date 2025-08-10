import math

import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- CARREGAR OS MODELOS ---
# 1. Classificadores Haar para encontrar o rosto e os olhos
face_cascade = cv2.CascadeClassifier('../classifier/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('../classifier/haarcascade_eye.xml')

# 2. Nosso modelo de Deep Learning treinado para estimar o olhar
gaze_model = load_model('../models/gaze_model.keras')

# Inicia a captura de vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inverte o quadro horizontalmente para um efeito de "espelho"
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Processa os dois primeiros olhos encontrados
        for i in range(min(2, len(eyes))):
            (ex, ey, ew, eh) = eyes[i]
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
            
            # ROI do olho
            eye_roi = roi_gray[ey:ey+eh, ex:ex+ew]
            
            # --- LÓGICA DE PREVISÃO COM O MODELO TREINADO ---
            # 1. Redimensiona a imagem do olho para o tamanho que o modelo espera (60x36)
            eye_roi_resized = cv2.resize(eye_roi, (60, 36))
            
            # 2. Pré-processa a imagem EXATAMENTE como fizemos no treinamento
            input_image = eye_roi_resized.astype(np.float32) / 255.0
            input_image = np.expand_dims(input_image, axis=-1) # Adiciona dimensão de canal
            input_image = np.expand_dims(input_image, axis=0)  # Adiciona dimensão de batch

            # 3. Faz a previsão!
            gaze_pred = gaze_model.predict(input_image)[0] # Pega o primeiro (e único) resultado
            pitch, yaw = gaze_pred
            
            # --- Desenha a Seta de Direção ---
            # Desenha a seta a partir do centro do olho
            start_point_on_eye = (x + ex + ew // 2, y + ey + eh // 2)
            arrow_length = 50
            end_x = int(start_point_on_eye[0] + arrow_length * math.sin(yaw))
            end_y = int(start_point_on_eye[1] + arrow_length * math.sin(pitch))
            
            cv2.arrowedLine(frame, start_point_on_eye, (end_x, end_y), (0, 0, 255), 2)

    cv2.imshow('Detector de Olhar com IA', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()