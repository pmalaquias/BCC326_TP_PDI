"""
WEBCAM ATTENTION DETECTOR - VERSÃO CORRIGIDA
============================================
Compatível com o modelo MPIIGaze melhorado

CORREÇÕES APLICADAS:
- Bug no load_metadata corrigido
- Threshold alinhado com treinamento
- Paths consistentes
- Melhor tratamento de erros
"""

import cv2
import numpy as np
import os
import urllib.request
import json
import time
import sys
from collections import deque
from tensorflow.keras.models import load_model

class ImprovedAttentionDetector:
    def __init__(self, model_path, metadata_path=None):
        """Inicializa detector melhorado"""
        print("🚀 Inicializando Detector de Atenção Melhorado...")
        
        # CORREÇÃO: Armazenar model_path primeiro
        self.model_path = model_path
        
        # Carregar metadados se disponível
        self.metadata = self.load_metadata(metadata_path)
        
        # Configurações baseadas nos metadados ou padrão
        self.IMG_SIZE = tuple(self.metadata.get('img_size', [64, 64]))
        self.TIME_WINDOW = self.metadata.get('time_window', 8)
        
        # CORREÇÃO: Usar threshold do treinamento ou padrão mais inteligente
        self.attention_threshold = 0.6
        
        print(f"  📋 Configurações:")
        print(f"    • Resolução: {self.IMG_SIZE}")
        print(f"    • Janela temporal: {self.TIME_WINDOW} frames")
        print(f"    • Threshold atenção: {self.attention_threshold:.3f}")
        
        # Carregar modelo
        print(f"  🧠 Carregando modelo: {model_path}")
        try:
            self.model = load_model(model_path)
            print("    ✅ Modelo carregado!")
        except Exception as e:
            print(f"    ❌ Erro ao carregar modelo: {e}")
            raise
        
        # Setup detecção facial
        self.setup_face_detection()
        
        # Buffers para análise temporal
        self.frame_buffer = deque(maxlen=self.TIME_WINDOW)
        self.prediction_buffer = deque(maxlen=30)  # Para suavização
        self.confidence_buffer = deque(maxlen=30)
        
        # Estatísticas
        self.total_predictions = 0
        self.attention_count = 0
        self.start_time = time.time()
        
        # Estado de qualidade
        self.face_quality_history = deque(maxlen=10)
        
        print("✅ Detector inicializado com sucesso!")
    
    def load_metadata(self, metadata_path):
        """CORREÇÃO: Carrega metadados do modelo de forma mais robusta"""
        
        # Tentar caminho fornecido
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"  📋 Metadados carregados: {metadata_path}")
                return metadata
            except Exception as e:
                print(f"  ⚠️ Erro ao carregar metadados fornecidos: {e}")
        
        # CORREÇÃO: Tentar encontrar no mesmo diretório do modelo
        if hasattr(self, 'model_path') and self.model_path:
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            metadata_path_auto = os.path.join(model_dir, 'model_metadata.json')
            
            if os.path.exists(metadata_path_auto):
                try:
                    with open(metadata_path_auto, 'r') as f:
                        metadata = json.load(f)
                    print(f"  📋 Metadados encontrados automaticamente: {metadata_path_auto}")
                    return metadata
                except Exception as e:
                    print(f"  ⚠️ Erro ao carregar metadados automáticos: {e}")
        
        # Tentar diretório padrão
        default_paths = [
            'model_metadata.json',
            '../models/model_metadata.json',
            'models/model_metadata.json'
        ]
        
        for path in default_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        metadata = json.load(f)
                    print(f"  📋 Metadados encontrados em: {path}")
                    return metadata
                except:
                    continue
        
        print("  ⚠️ Usando configurações padrão")
        return {}
    
    def setup_face_detection(self):
        """Configura detecção facial melhorada"""
        cascade_dir = os.path.join(os.path.dirname(__file__), 'haarcascades')
        os.makedirs(cascade_dir, exist_ok=True)
        
        # URLs dos classificadores
        cascades = {
            'face': {
                'url': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml',
                'file': 'haarcascade_frontalface_default.xml'
            },
            'eye': {
                'url': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_eye.xml',
                'file': 'haarcascade_eye.xml'
            }
        }
        
        # Baixar se necessário
        for name, info in cascades.items():
            file_path = os.path.join(cascade_dir, info['file'])
            
            if not os.path.exists(file_path):
                print(f"  📥 Baixando {name} cascade...")
                try:
                    urllib.request.urlretrieve(info['url'], file_path)
                    print(f"    ✅ {name} cascade baixado!")
                except Exception as e:
                    print(f"    ❌ Erro: {e}")
                    raise
        
        # Carregar classificadores
        face_path = os.path.join(cascade_dir, 'haarcascade_frontalface_default.xml')
        eye_path = os.path.join(cascade_dir, 'haarcascade_eye.xml')
        
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)
        
        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("❌ Erro ao carregar classificadores")
        
        print("  ✅ Detecção facial configurada!")
    
    def assess_face_quality(self, face_region, eyes):
        """Avalia qualidade da detecção facial"""
        quality_score = 0.0
        
        # Tamanho da face (faces maiores são melhores)
        face_area = face_region.shape[0] * face_region.shape[1]
        if face_area > 10000:  # 100x100 pixels
            quality_score += 0.3
        elif face_area > 5000:  # 70x70 pixels
            quality_score += 0.2
        
        # Detecção de olhos
        if len(eyes) >= 2:
            quality_score += 0.4
        elif len(eyes) == 1:
            quality_score += 0.2
        
        # Contraste da imagem
        contrast = np.std(face_region)
        if contrast > 20:
            quality_score += 0.2
        elif contrast > 10:
            quality_score += 0.1
        
        # Centralização da face
        frame_center_y = face_region.shape[0] // 2
        if 0.3 * face_region.shape[0] < frame_center_y < 0.7 * face_region.shape[0]:
            quality_score += 0.1
        
        return min(quality_score, 1.0)
    
    def extract_enhanced_face_region(self, frame, face):
        """Extrai região facial melhorada similar ao MPIIGaze"""
        x, y, w, h = face
        
        # Expandir região para incluir contexto (similar ao MPIIGaze)
        expansion_factor = 0.3
        margin_x = int(w * expansion_factor)
        margin_y = int(h * expansion_factor)
        
        # Coordenadas expandidas
        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(frame.shape[1], x + w + margin_x)
        y_end = min(frame.shape[0], y + h + margin_y)
        
        # Extrair região
        face_region = frame[y_start:y_end, x_start:x_end]
        
        # Verificar se região é válida
        if face_region.size == 0 or face_region.shape[0] < 32 or face_region.shape[1] < 32:
            return None, None, 0.0
        
        # Detectar olhos na região facial para qualidade
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY) if len(face_region.shape) == 3 else face_region
        eyes = self.eye_cascade.detectMultiScale(gray_face, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))
        
        # Avaliar qualidade
        quality = self.assess_face_quality(gray_face, eyes)
        
        return face_region, (x_start, y_start, x_end, y_end), quality
    
    def preprocess_face_enhanced(self, face_region):
        """CORREÇÃO: Preprocessamento EXATAMENTE alinhado com o treinamento"""
        if face_region is None or face_region.size == 0:
            return None
        
        try:
            # Converter para grayscale se necessário
            if len(face_region.shape) == 3:
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_region
            
            # Redimensionar
            face_resized = cv2.resize(face_gray, self.IMG_SIZE)
            
            # Converter para RGB (3 canais como no treinamento)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            
            # CORREÇÃO: Normalização EXATA do treinamento
            face_normalized = face_rgb.astype('float32') / 255.0
            
            # CORREÇÃO: Enhancement EXATO do treinamento
            face_enhanced = np.clip((face_normalized - 0.5) * 1.1 + 0.5, 0, 1)
            
            return face_enhanced
            
        except Exception as e:
            print(f"Erro no preprocessamento: {e}")
            return None
    
    def detect_attention_enhanced(self, frame):
        """Detecção de atenção melhorada"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detectar faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        if len(faces) == 0:
            return None, "❌ Nenhuma face detectada"
        
        # Usar maior face detectada
        face = max(faces, key=lambda f: f[2] * f[3])
        
        # Extrair região facial melhorada
        face_region, coords, quality = self.extract_enhanced_face_region(frame, face)
        
        if face_region is None:
            return None, "❌ Erro na extração facial"
        
        # Armazenar qualidade
        self.face_quality_history.append(quality)
        avg_quality = np.mean(list(self.face_quality_history))
        
        # Se qualidade muito baixa, reportar
        if quality < 0.3:
            return None, f"📉 Qualidade baixa: {quality:.2f}"
        
        # Preprocessar
        processed_frame = self.preprocess_face_enhanced(face_region)
        if processed_frame is None:
            return None, "❌ Erro no preprocessamento"
        
        # Adicionar ao buffer temporal
        self.frame_buffer.append(processed_frame)
        
        # Verificar se temos sequência completa
        if len(self.frame_buffer) < self.TIME_WINDOW:
            progress = len(self.frame_buffer) / self.TIME_WINDOW * 100
            return None, f"🔄 Coletando frames... ({progress:.0f}%)"
        
        # Fazer predição
        try:
            sequence = np.array(list(self.frame_buffer))
            sequence_batch = np.expand_dims(sequence, axis=0)

            prediction_raw = self.model.predict(sequence_batch, verbose=0)[0][0]
            print(f"Predição bruta: {prediction_raw:.4f}")  
            
            prediction_raw = self.model.predict(sequence_batch, verbose=0)[0][0]
            
            # CORREÇÃO: Aplicar threshold do treinamento
            is_attentive = prediction_raw > self.attention_threshold
            
            # Calcular confiança baseada na distância do threshold
            distance_from_threshold = abs(prediction_raw - self.attention_threshold)
            confidence = min(0.5 + distance_from_threshold, 1.0)
            
            # Adicionar aos buffers de suavização
            self.prediction_buffer.append(is_attentive)
            self.confidence_buffer.append(confidence)
            
            # Suavização temporal
            recent_predictions = list(self.prediction_buffer)[-10:]
            smoothed_attention = np.mean(recent_predictions) > 0.6
            avg_confidence = np.mean(list(self.confidence_buffer)[-10:])
            
            # Atualizar estatísticas
            self.total_predictions += 1
            if smoothed_attention:
                self.attention_count += 1
            
            return {
                'attention': smoothed_attention,
                'raw_attention': is_attentive,
                'confidence': float(avg_confidence),
                'prediction_value': float(prediction_raw),
                'face_coords': coords,
                'face_quality': float(quality),
                'avg_quality': float(avg_quality),
                'threshold_used': float(self.attention_threshold)
            }, None
            
        except Exception as e:
            return None, f"❌ Erro na predição: {str(e)}"
    
    def get_statistics(self):
        """Retorna estatísticas de uso"""
        elapsed_time = time.time() - self.start_time
        attention_percent = (self.attention_count / self.total_predictions * 100) if self.total_predictions > 0 else 0
        
        return {
            'total_predictions': self.total_predictions,
            'attention_count': self.attention_count,
            'attention_percentage': attention_percent,
            'elapsed_time': elapsed_time,
            'avg_face_quality': np.mean(list(self.face_quality_history)) if self.face_quality_history else 0
        }

def draw_enhanced_interface(frame, result, stats):
    """Desenha interface melhorada na tela"""
    if result:
        # Desenhar região facial
        coords = result['face_coords']
        x1, y1, x2, y2 = coords
        
        # Cor baseada na atenção
        color = (0, 255, 0) if result['attention'] else (0, 0, 255)
        thickness = 3 if result['attention'] else 2
        
        # Retângulo da face
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Status principal
        status = "ATENTO" if result['attention'] else "DISTRAÍDO"
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Fundo para texto
        text_size = cv2.getTextSize(status, font, 1.2, 2)[0]
        cv2.rectangle(frame, (x1, y1-40), (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(frame, status, (x1+5, y1-10), font, 1.2, (255, 255, 255), 2)
        
        # Informações detalhadas
        info_y = y1 + 25
        info_texts = [
            f"Conf: {result['confidence']:.1%}",
            f"Pred: {result['prediction_value']:.3f}",
            f"Qual: {result['face_quality']:.2f}"
        ]
        
        for i, text in enumerate(info_texts):
            cv2.putText(frame, text, (x1, info_y + i*20), font, 0.5, color, 1)
    
    # Estatísticas globais
    stats_y = 30
    stats_color = (255, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    stats_texts = [
        f"Predições: {stats['total_predictions']}",
        f"Atenção: {stats['attention_percentage']:.1f}%",
        f"Tempo: {stats['elapsed_time']:.0f}s",
        f"Qualidade Média: {stats['avg_face_quality']:.2f}"
    ]
    
    for i, text in enumerate(stats_texts):
        cv2.putText(frame, text, (10, stats_y + i*25), font, 0.6, stats_color, 1)

def print_enhanced_status(result, message, fps):
    """Status melhorado no terminal"""
    # Limpar linha
    sys.stdout.write('\r' + ' ' * 150)
    sys.stdout.write('\r')
    
    if result:
        # Emoji e status
        emoji = "🟢" if result['attention'] else "🔴"
        status = "ATENTO" if result['attention'] else "DISTRAÍDO"
        
        # Linha de status detalhada
        status_line = (
            f"{emoji} {status} | "
            f"Confiança: {result['confidence']:.1%} | "
            f"Predição: {result['prediction_value']:.4f} | "
            f"Threshold: {result['threshold_used']:.3f} | "
            f"Qualidade: {result['face_quality']:.2f} | "
            f"FPS: {fps:.1f}"
        )
    else:
        status_line = f"⏳ {message} | FPS: {fps:.1f}"
    
    sys.stdout.write(status_line)
    sys.stdout.flush()

def main():
    """Função principal melhorada"""
    
    # CORREÇÃO: Caminhos consistentes com o treinamento
    MODEL_PATH = os.path.join('..', 'models', 'mpiigaze_balanced_model.keras')
    METADATA_PATH = os.path.join('..', 'models', 'balanced_metadata.json')
    
    print("🎯 DETECTOR DE ATENÇÃO - COMPATÍVEL COM TREINAMENTO")
    print("=" * 60)
    
    # Verificar se modelo existe
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERRO: Modelo não encontrado em {MODEL_PATH}")
        print("Execute primeiro o script de treinamento para gerar o modelo.")
        
        # Tentar caminhos alternativos
        alternative_paths = [
            'mpiigaze_attention_model_improved.keras',
            '../mpiigaze_attention_model_improved.keras',
            'models/mpiigaze_attention_model_improved.keras'
        ]
        
        found = False
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                MODEL_PATH = alt_path
                METADATA_PATH = os.path.join(os.path.dirname(alt_path), 'model_metadata.json')
                print(f"✅ Modelo encontrado em: {MODEL_PATH}")
                found = True
                break
        
        if not found:
            return
    
    try:
        # Inicializar detector
        detector = ImprovedAttentionDetector(MODEL_PATH, METADATA_PATH)
    except Exception as e:
        print(f"❌ ERRO ao inicializar detector: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Verificar se metadados foram carregados corretamente
    if detector.metadata:
        print(f"\n📊 CONFIGURAÇÕES DO MODELO:")
        print(f"  • Threshold do treinamento: {detector.metadata.get('attention_threshold', 'N/A')}")
        print(f"  • Épocas treinadas: {detector.metadata.get('training_epochs', 'N/A')}")
        print(f"  • Accuracy final: {detector.metadata.get('final_val_accuracy', 'N/A')}")
        print(f"  • Total de sequências: {detector.metadata.get('total_sequences', 'N/A')}")
    
    # Configurar webcam
    print("\n🎥 Configurando webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ ERRO: Webcam não disponível")
        return
    
    # Configurações da webcam
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Webcam configurada!")
    
    # Interface informativa
    print("\n💡 CONTROLES:")
    print("  • Pressione 'q' para sair")
    print("  • Pressione 's' para salvar estatísticas")
    print("  • Pressione 'r' para resetar contadores")
    print("  • Pressione 'c' para calibrar threshold")
    print("=" * 60)
    print("\n🔄 Iniciando detecção...")
    
    # Variáveis de controle
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    save_counter = 0
    
    # Buffer para calibração
    calibration_predictions = deque(maxlen=100)
    calibration_mode = False
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\n❌ Erro ao capturar frame")
                break
            
            # Espelhar frame
            frame = cv2.flip(frame, 1)
            
            # Criar cópia para desenho
            display_frame = frame.copy()
            
            # Detectar atenção
            result, message = detector.detect_attention_enhanced(frame)
            
            # Obter estatísticas
            stats = detector.get_statistics()
            
            # Calcular FPS
            fps_counter += 1
            if fps_counter >= 30:
                fps_end_time = time.time()
                current_fps = fps_counter / (fps_end_time - fps_start_time)
                fps_counter = 0
                fps_start_time = fps_end_time
            
            # Desenhar interface
            draw_enhanced_interface(display_frame, result, stats)
            
            # Status no terminal
            print_enhanced_status(result, message, current_fps)
            
            # Adicionar à calibração se ativo
            if calibration_mode and result:
                calibration_predictions.append(result['prediction_value'])
            
            # Mostrar frame
            try:
                cv2.imshow('Detector de Atenção - Treinamento Compatível', display_frame)
                
                # Verificar teclas
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\n\n🛑 Encerrando por solicitação do usuário...")
                    break
                
                elif key == ord('s'):
                    # Salvar estatísticas
                    save_counter += 1
                    stats_filename = f"attention_stats_{save_counter}.json"
                    
                    detailed_stats = {
                        'timestamp': time.time(),
                        'session_stats': stats,
                        'model_config': {
                            'img_size': detector.IMG_SIZE,
                            'time_window': detector.TIME_WINDOW,
                            'threshold': detector.attention_threshold
                        },
                        'model_metadata': detector.metadata,
                        'recent_predictions': list(detector.prediction_buffer),
                        'recent_confidences': list(detector.confidence_buffer)
                    }
                    
                    try:
                        with open(stats_filename, 'w') as f:
                            json.dump(detailed_stats, f, indent=2)
                        print(f"\n💾 Estatísticas salvas em: {stats_filename}")
                    except Exception as e:
                        print(f"\n❌ Erro ao salvar: {e}")
                
                elif key == ord('r'):
                    # Resetar contadores
                    detector.total_predictions = 0
                    detector.attention_count = 0
                    detector.start_time = time.time()
                    detector.prediction_buffer.clear()
                    detector.confidence_buffer.clear()
                    print("\n🔄 Contadores resetados!")
                
                elif key == ord('c'):
                    # Modo calibração
                    if not calibration_mode:
                        calibration_mode = True
                        calibration_predictions.clear()
                        print("\n🎯 MODO CALIBRAÇÃO ATIVADO")
                        print("Olhe diretamente para a tela por ~10 segundos...")
                    else:
                        calibration_mode = False
                        if len(calibration_predictions) > 10:
                            # Calcular novo threshold baseado nas predições
                            predictions_array = np.array(calibration_predictions)
                            new_threshold = np.percentile(predictions_array, 25)  # 25º percentil
                            
                            print(f"\n📊 CALIBRAÇÃO CONCLUÍDA:")
                            print(f"  • Amostras coletadas: {len(calibration_predictions)}")
                            print(f"  • Threshold anterior: {detector.attention_threshold:.3f}")
                            print(f"  • Threshold sugerido: {new_threshold:.3f}")
                            
                            # Aplicar novo threshold
                            detector.attention_threshold = new_threshold
                            print(f"  ✅ Novo threshold aplicado: {new_threshold:.3f}")
                        else:
                            print("\n⚠️ Calibração cancelada - amostras insuficientes")
                
            except cv2.error:
                # Se não conseguir mostrar janela, continuar apenas com terminal
                pass
            
            # Pequena pausa para reduzir uso de CPU
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompido pelo usuário (Ctrl+C)")
    
    except Exception as e:
        print(f"\n❌ ERRO durante execução: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Estatísticas finais
        final_stats = detector.get_statistics()
        
        print("\n\n" + "="*60)
        print("📊 RELATÓRIO FINAL DA SESSÃO")
        print("="*60)
        print(f"⏱️  Duração da sessão: {final_stats['elapsed_time']:.1f} segundos")
        print(f"📈 Total de predições: {final_stats['total_predictions']}")
        print(f"🟢 Prestando atenção: {final_stats['attention_count']} ({final_stats['attention_percentage']:.1f}%)")
        print(f"🔴 Distraído: {final_stats['total_predictions'] - final_stats['attention_count']} ({100 - final_stats['attention_percentage']:.1f}%)")
        print(f"📊 Qualidade facial média: {final_stats['avg_face_quality']:.2f}")
        print(f"🎯 Threshold usado: {detector.attention_threshold:.3f}")
        
        # Análise da sessão
        if final_stats['total_predictions'] > 50:
            if final_stats['attention_percentage'] > 70:
                print("\n✅ SESSÃO: Alta concentração detectada!")
            elif final_stats['attention_percentage'] > 50:
                print("\n📊 SESSÃO: Concentração moderada")
            else:
                print("\n⚠️ SESSÃO: Baixa concentração - considere pausas")
        
        if final_stats['avg_face_quality'] < 0.5:
            print("\n💡 DICA: Melhore a iluminação ou posicionamento para melhor detecção")
        
        print("="*60)
        
        # Salvar log da sessão
        session_log = {
            'timestamp': time.time(),
            'duration_seconds': final_stats['elapsed_time'],
            'total_predictions': final_stats['total_predictions'],
            'attention_percentage': final_stats['attention_percentage'],
            'avg_face_quality': final_stats['avg_face_quality'],
            'threshold_used': detector.attention_threshold,
            'model_config': {
                'img_size': detector.IMG_SIZE,
                'time_window': detector.TIME_WINDOW
            },
            'model_metadata': detector.metadata
        }
        
        try:
            log_filename = f"session_log_{int(time.time())}.json"
            with open(log_filename, 'w') as f:
                json.dump(session_log, f, indent=2)
            print(f"📝 Log da sessão salvo em: {log_filename}")
        except Exception as e:
            print(f"⚠️ Erro ao salvar log: {e}")
        
        # Limpeza
        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass
        
        print("\n✅ Detector encerrado com sucesso!")

if __name__ == "__main__":
    main()