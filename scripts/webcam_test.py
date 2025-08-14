import cv2
import numpy as np
import os
import urllib.request
import json
import time
import sys
from collections import deque
from tensorflow.keras.models import load_model


class AttentionDetector:
    def __init__(self, model_path, metadata_path=None):
        self.model_path = model_path

        self.metadata = self.load_metadata(metadata_path)

        self.IMG_SIZE = tuple(self.metadata.get('img_size', [64, 64]))
        self.TIME_WINDOW = self.metadata.get('time_window', 8)
        self.attention_threshold = 0.6

        self.model = load_model(model_path)

        self.setup_face_detection()

        self.frame_buffer = deque(maxlen=self.TIME_WINDOW)
        self.prediction_buffer = deque(maxlen=30)
        self.confidence_buffer = deque(maxlen=30)

        self.total_predictions = 0
        self.attention_count = 0
        self.start_time = time.time()

    def load_metadata(self, metadata_path):
        if metadata_path and os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r') as f:
                    return json.load(f)
            except:
                pass

        if hasattr(self, 'model_path') and self.model_path:
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            metadata_path_auto = os.path.join(model_dir, 'model_metadata.json')

            if os.path.exists(metadata_path_auto):
                try:
                    with open(metadata_path_auto, 'r') as f:
                        return json.load(f)
                except:
                    pass

        default_paths = [
            'model_metadata.json',
            '../models/model_metadata.json',
            'models/model_metadata.json'
        ]

        for path in default_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        return json.load(f)
                except:
                    continue

        return {}

    def setup_face_detection(self):
        cascade_dir = os.path.join(os.path.dirname(__file__), 'haarcascades')
        os.makedirs(cascade_dir, exist_ok=True)

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

        for name, info in cascades.items():
            file_path = os.path.join(cascade_dir, info['file'])
            if not os.path.exists(file_path):
                urllib.request.urlretrieve(info['url'], file_path)

        face_path = os.path.join(
            cascade_dir, 'haarcascade_frontalface_default.xml')
        eye_path = os.path.join(cascade_dir, 'haarcascade_eye.xml')

        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

        if self.face_cascade.empty() or self.eye_cascade.empty():
            raise RuntimeError("Erro ao carregar classificadores")

    def extract_face_region(self, frame, face):
        x, y, w, h = face

        expansion_factor = 0.3
        margin_x = int(w * expansion_factor)
        margin_y = int(h * expansion_factor)

        x_start = max(0, x - margin_x)
        y_start = max(0, y - margin_y)
        x_end = min(frame.shape[1], x + w + margin_x)
        y_end = min(frame.shape[0], y + h + margin_y)

        face_region = frame[y_start:y_end, x_start:x_end]

        if face_region.size == 0 or face_region.shape[0] < 32 or face_region.shape[1] < 32:
            return None, None

        return face_region, (x_start, y_start, x_end, y_end)

    def preprocess_face(self, face_region):
        if face_region is None or face_region.size == 0:
            return None

        try:
            if len(face_region.shape) == 3:
                face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            else:
                face_gray = face_region

            face_resized = cv2.resize(face_gray, self.IMG_SIZE)
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)
            face_normalized = face_rgb.astype('float32') / 255.0
            face_enhanced = np.clip((face_normalized - 0.5) * 1.1 + 0.5, 0, 1)

            return face_enhanced
        except:
            return None

    def detect_attention(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None, "Nenhuma face detectada"

        face = max(faces, key=lambda f: f[2] * f[3])
        face_region, coords = self.extract_face_region(frame, face)

        if face_region is None:
            return None, "Erro na extração facial"

        processed_frame = self.preprocess_face(face_region)
        if processed_frame is None:
            return None, "Erro no preprocessamento"

        self.frame_buffer.append(processed_frame)

        if len(self.frame_buffer) < self.TIME_WINDOW:
            progress = len(self.frame_buffer) / self.TIME_WINDOW * 100
            return None, f"Coletando frames... ({progress:.0f}%)"

        try:
            sequence = np.array(list(self.frame_buffer))
            sequence_batch = np.expand_dims(sequence, axis=0)

            prediction_raw = self.model.predict(
                sequence_batch, verbose=0)[0][0]
            is_attentive = prediction_raw > self.attention_threshold

            distance_from_threshold = abs(
                prediction_raw - self.attention_threshold)
            confidence = min(0.5 + distance_from_threshold, 1.0)

            self.prediction_buffer.append(is_attentive)
            self.confidence_buffer.append(confidence)

            recent_predictions = list(self.prediction_buffer)[-10:]
            smoothed_attention = np.mean(recent_predictions) > 0.6
            avg_confidence = np.mean(list(self.confidence_buffer)[-10:])

            self.total_predictions += 1
            if smoothed_attention:
                self.attention_count += 1

            return {
                'attention': smoothed_attention,
                'confidence': float(avg_confidence),
                'face_coords': coords
            }, None

        except Exception as e:
            return None, f"Erro na predição: {str(e)}"

    def get_statistics(self):
        elapsed_time = time.time() - self.start_time
        attention_percent = (self.attention_count / self.total_predictions *
                             100) if self.total_predictions > 0 else 0

        return {
            'total_predictions': self.total_predictions,
            'attention_count': self.attention_count,
            'attention_percentage': attention_percent,
            'elapsed_time': elapsed_time
        }


def draw_interface(frame, result):
    if result:
        coords = result['face_coords']
        x1, y1, x2, y2 = coords

        color = (0, 255, 0) if result['attention'] else (0, 0, 255)
        thickness = 3 if result['attention'] else 2

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        status = "ATENTO" if result['attention'] else "DISTRAIDO"
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size = cv2.getTextSize(status, font, 1.2, 2)[0]
        cv2.rectangle(frame, (x1, y1-40),
                      (x1 + text_size[0] + 10, y1), color, -1)
        cv2.putText(frame, status, (x1+5, y1-10),
                    font, 1.2, (255, 255, 255), 2)

        cv2.putText(frame, f"Confiabilidade: {result['confidence']:.1%}",
                    (x1, y1 + 25), font, 0.5, color, 1)


def print_status(result, message):
    sys.stdout.write('\r' + ' ' * 100)
    sys.stdout.write('\r')

    if result:
        emoji = "🟢" if result['attention'] else "🔴"
        status = "ATENTO" if result['attention'] else "DISTRAIDO"
        status_line = f"{emoji} {status} | Confiabilidade: {result['confidence']:.1%}"
    else:
        status_line = f"⏳ {message}"

    sys.stdout.write(status_line)
    sys.stdout.flush()


def main():
    MODEL_PATH = os.path.join('..', 'models', 'mpiigaze_balanced_model.keras')
    METADATA_PATH = os.path.join('..', 'models', 'balanced_metadata.json')

    print("DETECTOR DE ATENÇÃO")
    print("=" * 40)

    if not os.path.exists(MODEL_PATH):
        print(f"Erro: Modelo não encontrado em {MODEL_PATH}")

        alternative_paths = [
            'mpiigaze_attention_model_improved.keras',
            '../mpiigaze_attention_model_improved.keras',
            'models/mpiigaze_attention_model_improved.keras'
        ]

        found = False
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                MODEL_PATH = alt_path
                METADATA_PATH = os.path.join(
                    os.path.dirname(alt_path), 'model_metadata.json')
                found = True
                break

        if not found:
            return

    try:
        detector = AttentionDetector(MODEL_PATH, METADATA_PATH)
    except Exception as e:
        print(f"Erro ao inicializar detector: {e}")
        return

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Webcam não disponível")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("\nControles:")
    print("  • Pressione 'q' para sair")
    print("=" * 40)
    print("\nIniciando detecção...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nErro ao capturar frame")
                break

            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()

            result, message = detector.detect_attention(frame)

            draw_interface(display_frame, result)
            print_status(result, message)

            try:
                cv2.imshow('Detector de Atenção', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n\nEncerrando...")
                    break
            except cv2.error:
                pass

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n\nInterrompido pelo usuário")

    except Exception as e:
        print(f"\nErro durante execução: {e}")

    finally:
        final_stats = detector.get_statistics()

        print("\n\n" + "="*60)
        print("📊 RELATÓRIO FINAL DA SESSÃO")
        print("="*60)
        print(
            f"⏱️  Duração da sessão: {final_stats['elapsed_time']:.1f} segundos")
        print(f"📈 Total de predições: {final_stats['total_predictions']}")
        print(
            f"🟢 Prestando atenção: {final_stats['attention_count']} ({final_stats['attention_percentage']:.1f}%)")
        print(
            f"🔴 Distraído: {final_stats['total_predictions'] - final_stats['attention_count']} ({100 - final_stats['attention_percentage']:.1f}%)")

        # Análise da sessão
        if final_stats['total_predictions'] > 50:
            if final_stats['attention_percentage'] > 70:
                print("\n✅ SESSÃO: Alta concentração detectada!")
                print("💡 Parabéns! Você manteve foco excelente durante a sessão.")
            elif final_stats['attention_percentage'] > 50:
                print("\n📊 SESSÃO: Concentração moderada")
                print("💡 Bom trabalho! Considere pequenas pausas para manter o foco.")
            else:
                print("\n⚠️ SESSÃO: Baixa concentração detectada")
                print(
                    "💡 DICA: Tente fazer pausas regulares.")
        else:
            print("\n📝 SESSÃO: Dados insuficientes para análise completa")

        print("="*60)

        cap.release()
        try:
            cv2.destroyAllWindows()
        except:
            pass

        print("\nDetector encerrado!")


if __name__ == "__main__":
    main()
