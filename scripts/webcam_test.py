"""
WEBCAM ATTENTION DETECTOR - CORRIGIDO PARA MPIIGAZE
==================================================
Ajustado para extrair faixas faciais compatíveis com MPIIGaze

Dependências: tensorflow numpy opencv-python
"""

import cv2
import numpy as np
import os
import urllib.request
from tensorflow.keras.models import load_model
from collections import deque
import time
import sys

class AttentionDetectorMPIIGazeCompatible:
    def __init__(self, model_path):
        """Inicializa detector compatível com MPIIGaze"""
        print("Inicializando Detector de Atenção (Compatível MPIIGaze)...")
        
        # Configurações
        self.IMG_SIZE = (64, 64)
        self.TIME_WINDOW = 10
        
        # Carregar modelo
        print(f"Carregando modelo: {model_path}")
        self.model = load_model(model_path)
        print("✅ Modelo carregado com sucesso!")
        
        # Setup haarcascades
        self.setup_haarcascades()
        
        # Buffer temporal
        self.eye_buffer = deque(maxlen=self.TIME_WINDOW)
        
        # Estatísticas
        self.total_predictions = 0
        self.attention_count = 0
        self.start_time = time.time()
        
        print("✅ Detector inicializado!")
    
    def setup_haarcascades(self):
        """Configura haarcascades"""
        cascade_dir = os.path.join(os.path.dirname(__file__), 'haarcascades')
        os.makedirs(cascade_dir, exist_ok=True)
        
        cascades = {
            'face': {
                'url': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
                'file': 'haarcascade_frontalface_default.xml'
            }
        }
        
        for cascade_name, info in cascades.items():
            file_path = os.path.join(cascade_dir, info['file'])
            
            if not os.path.exists(file_path):
                print(f"📥 Baixando {cascade_name} cascade...")
                try:
                    urllib.request.urlretrieve(info['url'], file_path)
                    print(f"✅ {cascade_name} cascade baixado!")
                except Exception as e:
                    print(f"❌ Erro ao baixar {cascade_name}: {e}")
                    raise
        
        face_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
        self.face_cascade = cv2.CascadeClassifier(face_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("❌ Erro ao carregar haarcascades")
        
        print("✅ Haarcascades configurados!")
    
    def extract_mpiigaze_compatible_strip(self, frame, face):
        """
        Extrai faixa facial COMPATÍVEL com MPIIGaze
        Faixa horizontal ampla incluindo olhos, sobrancelhas e contexto
        """
        x, y, w, h = face
        
        # Parâmetros para faixa similar ao MPIIGaze
        # Baseado nas suas imagens 3-4: faixa horizontal ampla
        strip_height = int(h * 0.6)  # 60% da altura da face (mais ampla)
        strip_y_start = y + int(h * 0.15)  # Começar mais alto (incluir sobrancelhas)
        strip_y_end = strip_y_start + strip_height
        
        # Manter largura completa da face + margens
        margin_x = int(w * 0.15)  # 15% de margem lateral
        strip_x_start = max(0, x - margin_x)
        strip_x_end = min(frame.shape[1], x + w + margin_x)
        
        # Garantir limites da imagem
        strip_y_start = max(0, strip_y_start)
        strip_y_end = min(frame.shape[0], strip_y_end)
        
        # Extrair faixa facial
        facial_strip = frame[strip_y_start:strip_y_end, strip_x_start:strip_x_end]
        
        return facial_strip, (strip_x_start, strip_y_start, strip_x_end, strip_y_end)
    
    def preprocess_facial_strip(self, facial_strip):
        """
        Preprocessa faixa facial para o formato do modelo
        COMPATÍVEL com o preprocessing do MPIIGaze
        """
        if facial_strip.size == 0:
            return None
        
        # Converter para grayscale primeiro (como MPIIGaze)
        if len(facial_strip.shape) == 3:
            facial_strip_gray = cv2.cvtColor(facial_strip, cv2.COLOR_BGR2GRAY)
        else:
            facial_strip_gray = facial_strip
        
        # Redimensionar para 64x64
        strip_resized = cv2.resize(facial_strip_gray, self.IMG_SIZE)
        
        # Converter de volta para RGB (3 canais) como esperado pelo modelo
        strip_rgb = cv2.cvtColor(strip_resized, cv2.COLOR_GRAY2RGB)
        
        # Normalizar para [0,1]
        strip_normalized = strip_rgb.astype('float32') / 255.0
        
        return strip_normalized
    
    def detect_attention(self, frame):
        """Detecta atenção usando faixa facial compatível com MPIIGaze"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(100, 100)
        )
        
        if len(faces) == 0:
            return None, "❌ Nenhuma face detectada"
        
        # Usar primeira face detectada
        face = faces[0]
        
        # Extrair faixa facial compatível com MPIIGaze
        facial_strip, strip_coords = self.extract_mpiigaze_compatible_strip(frame, face)
        
        if facial_strip.size == 0:
            return None, "👁️ Erro ao extrair faixa facial"
        
        # DEBUG: Salvar algumas amostras para comparação
        if self.total_predictions < 10:
            debug_dir = "debug_mpiigaze_compatible"
            os.makedirs(debug_dir, exist_ok=True)
            
            # Salvar faixa original
            cv2.imwrite(f"{debug_dir}/strip_original_{self.total_predictions}.jpg", facial_strip)
            
            # Salvar faixa processada
            strip_processed = self.preprocess_facial_strip(facial_strip)
            if strip_processed is not None:
                strip_display = (strip_processed * 255).astype(np.uint8)
                strip_display_bgr = cv2.cvtColor(strip_display, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f"{debug_dir}/strip_processed_{self.total_predictions}.jpg", strip_display_bgr)
            
            print(f"\n💾 Debug: Salvou faixa compatível {self.total_predictions}")
            print(f"   Dimensões originais: {facial_strip.shape}")
            print(f"   Coordenadas: {strip_coords}")
        
        # Preprocessar
        try:
            strip_processed = self.preprocess_facial_strip(facial_strip)
            if strip_processed is None:
                return None, "❌ Erro no preprocessing"
        except Exception as e:
            return None, f"❌ Erro preprocessing: {str(e)}"
        
        # Adicionar ao buffer temporal
        self.eye_buffer.append(strip_processed)
        
        # Se temos sequência completa, fazer predição
        if len(self.eye_buffer) == self.TIME_WINDOW:
            sequence = np.array(list(self.eye_buffer))
            sequence = np.expand_dims(sequence, axis=0)  # Batch dimension
            
            try:
                prediction = self.model.predict(sequence, verbose=0)[0][0]
                
                # DEBUG: Mostrar detalhes da predição
                if self.total_predictions < 5:
                    print(f"\n🔍 Debug predição {self.total_predictions}:")
                    print(f"   Shape da sequência: {sequence.shape}")
                    print(f"   Min/Max valores: {sequence.min():.3f}/{sequence.max():.3f}")
                    print(f"   Valor predição bruta: {prediction}")
                
                # Threshold para classificação
                is_attentive = prediction > 0.5
                confidence = prediction if is_attentive else (1 - prediction)
                
                self.total_predictions += 1
                if is_attentive:
                    self.attention_count += 1
                
                return {
                    'attention': is_attentive,
                    'confidence': float(confidence),
                    'prediction_value': float(prediction),
                    'strip_coords': strip_coords
                }, None
                
            except Exception as e:
                return None, f"❌ Erro na predição: {str(e)}"
        
        return None, f"🔄 Coletando frames... ({len(self.eye_buffer)}/{self.TIME_WINDOW})"
    
    def print_status(self, result, message, fps):
        """Imprime status no terminal"""
        # Limpar linha anterior
        sys.stdout.write('\r' + ' ' * 140)
        sys.stdout.write('\r')
        
        if result:
            # Status principal
            status_emoji = "🟢" if result['attention'] else "🔴"
            status_text = "PRESTANDO ATENÇÃO" if result['attention'] else "DISTRAÍDO"
            
            # Estatísticas
            elapsed_time = time.time() - self.start_time
            attention_percent = (self.attention_count / self.total_predictions * 100) if self.total_predictions > 0 else 0
            
            # Linha de status completa
            status_line = (
                f"{status_emoji} {status_text} | "
                f"Confiança: {result['confidence']:.1%} | "
                f"Predição: {result['prediction_value']:.6f} | "
                f"FPS: {fps:.1f} | "
                f"Tempo: {elapsed_time:.0f}s | "
                f"Atenção: {attention_percent:.1f}% ({self.attention_count}/{self.total_predictions})"
            )
        else:
            # Mensagem de status
            elapsed_time = time.time() - self.start_time
            status_line = f"⏳ {message} | Tempo: {elapsed_time:.0f}s | FPS: {fps:.1f}"
        
        sys.stdout.write(status_line)
        sys.stdout.flush()

def draw_facial_strip_region(frame, result):
    """Desenha a região da faixa facial extraída"""
    if result and 'strip_coords' in result:
        x1, y1, x2, y2 = result['strip_coords']
        color = (0, 255, 0) if result['attention'] else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Adicionar texto indicativo
        cv2.putText(frame, "Faixa MPIIGaze-Compatible", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def main():
    """Função principal"""
    
    MODEL_PATH = os.path.join('..', 'models', 'mpiigaze_attention_model_normalized.keras')
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        return
    
    try:
        detector = AttentionDetectorMPIIGazeCompatible(MODEL_PATH)
    except Exception as e:
        print(f"❌ ERRO ao inicializar: {e}")
        return
    
    print("Iniciando webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERRO: Webcam não disponível")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("✅ Sistema pronto!")
    print("\n" + "="*80)
    print("🎥 DETECTOR DE ATENÇÃO - COMPATÍVEL COM MPIIGAZE")
    print("="*80)
    print("• Extrai faixa horizontal AMPLA como MPIIGaze")
    print("• Inclui olhos, sobrancelhas e contexto")
    print("• Salva primeiras 10 faixas em debug_mpiigaze_compatible/")
    print("• Compare com as imagens do MPIIGaze!")
    print("• Pressione Ctrl+C para sair")
    print("="*80)
    print("\nStatus em tempo real:")
    print()
    
    # Variáveis FPS
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Espelhar frame
            frame = cv2.flip(frame, 1)
            
            # Fazer uma cópia para desenhar
            display_frame = frame.copy()
            
            # Detectar atenção
            result, message = detector.detect_attention(frame)
            
            # Desenhar região extraída
            draw_facial_strip_region(display_frame, result)
            
            # Calcular FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                current_fps = fps_counter / (fps_end_time - fps_start_time)
                fps_counter = 0
                fps_start_time = fps_end_time
            
            # Mostrar status no terminal
            detector.print_status(result, message, current_fps)
            
            # Mostrar frame com indicação visual
            try:
                cv2.imshow('Detector - MPIIGaze Compatible', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except:
                # Se não conseguir mostrar janela, continue apenas com terminal
                pass
            
            # Pequena pausa
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompido pelo usuário")
        
        # Mostrar estatísticas finais
        elapsed_time = time.time() - detector.start_time
        attention_percent = (detector.attention_count / detector.total_predictions * 100) if detector.total_predictions > 0 else 0
        
        print("\n" + "="*50)
        print("📊 ESTATÍSTICAS FINAIS")
        print("="*50)
        print(f"⏱️ Tempo total: {elapsed_time:.1f} segundos")
        print(f"📈 Total de predições: {detector.total_predictions}")
        print(f"🟢 Prestando atenção: {detector.attention_count} ({attention_percent:.1f}%)")
        print(f"🔴 Distraído: {detector.total_predictions - detector.attention_count} ({100-attention_percent:.1f}%)")
        print("="*50)
        print("\n💾 Compare as faixas em debug_mpiigaze_compatible/ com as do MPIIGaze!")
        
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        print("✅ Encerrado!")

if __name__ == "__main__":
    main()