import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

class EyeDetector:
    def __init__(self):
        # Carrega os classificadores Haar Cascade pré-treinados
        self.face_cascade = cv2.CascadeClassifier('classifier/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('classifier/haarcascade_eye.xml')
        
        # Verifica se os classificadores foram carregados corretamente
        if self.face_cascade.empty():
            print("Erro: Não foi possível carregar o classificador de faces")
        if self.eye_cascade.empty():
            print("Erro: Não foi possível carregar o classificador de olhos")
        
        # Carrega o modelo de atenção do olhar
        try:
            self.gaze_model = keras.models.load_model('models/gaze_attention_model.keras')
            print("Modelo de atenção do olhar carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar modelo de atenção: {e}")
            self.gaze_model = None
    
    def detect_eyes(self, frame):
        """Detecta faces e olhos no frame"""
        # Converte para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detecta faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        eye_regions = []
        combined_eye_region = None
        
        for (x, y, w, h) in faces:
            # Região de interesse para os olhos (parte superior da face)
            roi_gray = gray[y:y + int(h * 0.6), x:x + w]
            roi_color = frame[y:y + int(h * 0.6), x:x + w]
            
            # Detecta olhos na região de interesse
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(20, 20)
            )
            
            # Desenha retângulo ao redor da face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Processa cada olho detectado
            for (ex, ey, ew, eh) in eyes:
                # Coordenadas absolutas do olho
                eye_x = x + ex
                eye_y = y + ey
                
                # Desenha retângulo ao redor de cada olho
                cv2.rectangle(frame, (eye_x, eye_y), (eye_x + ew, eye_y + eh), (0, 255, 0), 2)
                
                # Adiciona à lista de regiões dos olhos
                eye_regions.append((eye_x, eye_y, ew, eh))
            
            # Se detectou pelo menos 2 olhos, cria bounding box combinada
            if len(eye_regions) >= 2:
                # Encontra as coordenadas mínimas e máximas para incluir ambos os olhos
                min_x = min(eye_regions, key=lambda x: x[0])[0]
                min_y = min(eye_regions, key=lambda x: x[1])[1]
                max_x = max(eye_regions, key=lambda x: x[0] + x[2])[0] + max(eye_regions, key=lambda x: x[0] + x[2])[2]
                max_y = max(eye_regions, key=lambda x: x[1] + x[3])[1] + max(eye_regions, key=lambda x: x[1] + x[3])[3]
                
                # Adiciona margem para melhor visualização
                margin = 20
                min_x = max(0, min_x - margin)
                min_y = max(0, min_y - margin)
                max_x = min(frame.shape[1], max_x + margin)
                max_y = min(frame.shape[0], max_y + margin)
                
                # Desenha retângulo combinado ao redor dos dois olhos
                cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                
                # Adiciona texto "Região dos Olhos"
                cv2.putText(frame, "Regiao dos Olhos", (min_x, min_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                combined_eye_region = (min_x, min_y, max_x - min_x, max_y - min_y)
                
                # Faz predição de atenção usando a região combinada dos olhos
                attention_result = self.predict_attention(combined_eye_region, frame)
                
                # Adiciona texto da predição de atenção na bounding box da face
                if attention_result is not None and len(attention_result) == 2:
                    attention_text, attention_score = attention_result
                    if attention_text != "Modelo não disponível" and attention_text != "Erro na predição":
                        # Posiciona o texto acima da face
                        text_x = x
                        text_y = max(0, y - 10)
                        
                        # Escolhe cor baseada no resultado
                        if attention_text == "ATENÇÃO":
                            color = (0, 255, 0)  # Verde para atenção
                        else:
                            color = (0, 0, 255)  # Vermelho para sem atenção
                        
                        # Adiciona texto da predição
                        cv2.putText(frame, f"{attention_text}: {attention_score:.2f}", 
                                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return eye_regions, combined_eye_region
    
    def predict_attention(self, eye_region, frame):
        """Prediz se há atenção baseada na região dos olhos"""
        if self.gaze_model is None or eye_region is None:
            return "Modelo não disponível"
        
        try:
            x, y, w, h = eye_region
            
            # Extrai a região dos olhos do frame
            eye_img = frame[y:y+h, x:x+w]
            
            # Redimensiona para o tamanho esperado pelo modelo (assumindo 64x64)
            eye_img_resized = cv2.resize(eye_img, (64, 64))
            
            # Converte para escala de cinza se necessário
            if len(eye_img_resized.shape) == 3:
                eye_img_gray = cv2.cvtColor(eye_img_resized, cv2.COLOR_BGR2GRAY)
            else:
                eye_img_gray = eye_img_resized
            
            # Normaliza os valores para [0, 1]
            eye_img_normalized = eye_img_gray.astype(np.float32) / 255.0
            
            # Adiciona dimensões de batch e canal se necessário
            if len(eye_img_normalized.shape) == 2:
                eye_img_input = eye_img_normalized.reshape(1, 64, 64, 1)
            else:
                eye_img_input = eye_img_normalized.reshape(1, 64, 64, 1)
            
            # Faz a predição
            prediction = self.gaze_model.predict(eye_img_input, verbose=0)
            
            # Interpreta a predição (assumindo que é binária: 0 = sem atenção, 1 = com atenção)
            if len(prediction.shape) > 1:
                prediction = prediction.flatten()
            
            attention_score = prediction[0]
            
            # Determina o resultado baseado no score
            if attention_score > 0.5:
                return "ATENÇÃO", attention_score
            else:
                return "SEM ATENÇÃO", attention_score
                
        except Exception as e:
            print(f"Erro na predição: {e}")
            return "Erro na predição", 0.0
    
    def run_webcam(self):
        """Executa a detecção em tempo real usando webcam"""
        # Inicializa a webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Erro: Não foi possível abrir a webcam")
            return
        
        print("Pressione 'q' para sair")
        print("Pressione 's' para salvar uma captura")
        
        while True:
            # Captura frame da webcam
            ret, frame = cap.read()
            
            if not ret:
                print("Erro: Não foi possível capturar frame da webcam")
                break
            
            # Detecta olhos
            eye_regions, combined_region = self.detect_eyes(frame)
            
            # Adiciona informações na tela
            cv2.putText(frame, f"Olhos detectados: {len(eye_regions)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostra o frame
            cv2.imshow('Detecção de Olhos - Webcam', frame)
            
            # Aguarda tecla
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Salva a captura
                filename = f"captura_olhos_{len(os.listdir('imgs')) + 1}.png"
                cv2.imwrite(f"imgs/{filename}", frame)
                print(f"Captura salva como: {filename}")
        
        # Libera recursos
        cap.release()
        cv2.destroyAllWindows()

def main():
    """Função principal"""
    print("Iniciando sistema de detecção de olhos...")
    
    # Cria diretório de imagens se não existir
    if not os.path.exists('imgs'):
        os.makedirs('imgs')
    
    # Inicializa o detector
    detector = EyeDetector()
    
    try:
        # Executa a detecção em tempo real
        detector.run_webcam()
    except KeyboardInterrupt:
        print("\nPrograma interrompido pelo usuário")
    except Exception as e:
        print(f"Erro: {e}")
    finally:
        print("Programa finalizado")

if __name__ == "__main__":
    main()
