import os
import numpy as np
import cv2
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import shutil
import json

# --- CONFIGURAÇÕES ---
CAMINHO_BASE = '..'
MPIIGAZE_PATH = os.path.join(CAMINHO_BASE, 'mpiigaze_real', 'MPIIGaze', 'Data', 'Normalized')
output_dir_sequences = 'processed_sequences_balanced'
MODELO_SALVO = os.path.join(CAMINHO_BASE, 'models', 'mpiigaze_balanced_model.keras')

IMG_SIZE = (32, 32)
TIME_WINDOW = 4
TARGET_FILES = 30

def preprocess_simple(img):
    img_resized = cv2.resize(img, IMG_SIZE)
    if len(img_resized.shape) == 2:
        img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    return img_resized.astype('float32') / 255.0

def create_simple_model(input_shape):
    model = Sequential([
        TimeDistributed(Conv2D(16, (5, 5), activation='relu'), input_shape=input_shape),
        TimeDistributed(MaxPooling2D((4, 4))),
        TimeDistributed(Flatten()),
        LSTM(32),
        Dense(1, activation='sigmoid')
    ])
    return model

# --- CARREGAMENTO ---
print("🔄 Carregamento balanceado forçado...")

if os.path.exists(output_dir_sequences):
    shutil.rmtree(output_dir_sequences)
os.makedirs(output_dir_sequences)

all_images = []
all_angles = []
processed_files = 0

for participant_dir in sorted(os.listdir(MPIIGAZE_PATH)):
    if processed_files >= TARGET_FILES:
        break
    
    participant_path = os.path.join(MPIIGAZE_PATH, participant_dir)
    if not os.path.isdir(participant_path):
        continue
    
    for day_file in sorted(os.listdir(participant_path)):
        if processed_files >= TARGET_FILES:
            break
        if not day_file.endswith('.mat'):
            continue
        
        mat_path = os.path.join(participant_path, day_file)
        try:
            mat_data = loadmat(mat_path)
            data = mat_data['data'][0, 0]
            eye_data = data['right'][0, 0]
            images = eye_data['image']
            gazes = eye_data['gaze']
            
            for img, gaze in zip(images, gazes):
                img_processed = preprocess_simple(img)
                pitch, yaw = gaze[0], gaze[1]
                angle = np.sqrt(pitch**2 + yaw**2)
                
                all_images.append(img_processed)
                all_angles.append(angle)
            
            processed_files += 1
            print(f"  {processed_files} arquivos, {len(all_images)} imagens")
                
        except Exception as e:
            continue

print(f"Total: {len(all_images)} imagens")

# --- BALANCEAMENTO FORÇADO ---
print("🎯 Forçando balanceamento 50-50...")

angles_array = np.array(all_angles)
images_array = np.array(all_images)

# Usar mediana para dividir 50-50
median_angle = np.median(angles_array)
print(f"Mediana dos ângulos: {median_angle:.3f}")

# Criar labels balanceados
labels = []
for angle in all_angles:
    label = 1 if angle <= median_angle else 0
    labels.append(label)

labels_array = np.array(labels)
unique, counts = np.unique(labels_array, return_counts=True)
print(f"Distribuição FORÇADA: {dict(zip(unique, counts))}")

# --- SEQUÊNCIAS BALANCEADAS ---
print("🔄 Criando sequências balanceadas...")

sequences = []
seq_labels = []

for i in range(0, len(images_array) - TIME_WINDOW + 1, TIME_WINDOW//2):  # 50% overlap
    sequence = images_array[i:i + TIME_WINDOW]
    window_labels = labels_array[i:i + TIME_WINDOW]
    
    # Votação: maioria decide
    seq_label = 1 if np.sum(window_labels) >= TIME_WINDOW//2 else 0
    
    sequences.append(sequence)
    seq_labels.append(seq_label)

X = np.array(sequences)
y = np.array(seq_labels)

print(f"Sequências: {len(X)}")
seq_unique, seq_counts = np.unique(y, return_counts=True)
print(f"Distribuição sequências: {dict(zip(seq_unique, seq_counts))}")

# Se ainda não estiver balanceado, balancear manualmente
if len(seq_unique) == 2:
    min_class = np.argmin(seq_counts)
    min_count = seq_counts[min_class]
    
    # Pegar amostras iguais de cada classe
    class_0_indices = np.where(y == 0)[0]
    class_1_indices = np.where(y == 1)[0]
    
    # Limitar ao menor número
    samples_per_class = min(len(class_0_indices), len(class_1_indices), 500)
    
    # Selecionar amostras aleatórias
    np.random.seed(42)
    selected_0 = np.random.choice(class_0_indices, samples_per_class, replace=False)
    selected_1 = np.random.choice(class_1_indices, samples_per_class, replace=False)
    
    # Combinar
    selected_indices = np.concatenate([selected_0, selected_1])
    np.random.shuffle(selected_indices)
    
    X = X[selected_indices]
    y = y[selected_indices]
    
    print(f"BALANCEADO FINAL: {len(X)} sequências")
    print(f"Distribuição final: {dict(zip(*np.unique(y, return_counts=True)))}")

# --- DIVISÃO ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Treino: {len(X_train)}, Teste: {len(X_test)}")
print(f"Treino dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
print(f"Teste dist: {dict(zip(*np.unique(y_test, return_counts=True)))}")

# --- MODELO ---
print("🚀 Modelo balanceado...")

model = create_simple_model(X_train.shape[1:])
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- TREINAMENTO ---
print("🎯 Treinamento balanceado...")

history = model.fit(
    X_train, y_train,
    batch_size=16,  # Menor para treinar mais devagar
    epochs=30,      # Mais épocas
    validation_data=(X_test, y_test),
    verbose=1
)

# --- AVALIAÇÃO ---
print("📊 Avaliação...")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Accuracy teste: {test_acc:.3f}")

y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_binary))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_binary))

# Testar diferentes thresholds
print("\nTeste de thresholds:")
for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
    y_pred_thresh = (y_pred > thresh).astype(int)
    acc = np.mean(y_test == y_pred_thresh.flatten())
    print(f"Threshold {thresh}: Accuracy = {acc:.3f}")

# --- SALVAR ---
os.makedirs(os.path.dirname(MODELO_SALVO), exist_ok=True)
model.save(MODELO_SALVO)

metadata = {
    'model_path': MODELO_SALVO,
    'img_size': IMG_SIZE,
    'time_window': TIME_WINDOW,
    'test_accuracy': float(test_acc),
    'attention_threshold': 0.5,
    'median_angle_used': float(median_angle)
}

metadata_path = os.path.join(os.path.dirname(MODELO_SALVO), 'balanced_metadata.json')
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Modelo balanceado salvo: {MODELO_SALVO}")
print(f"Mediana usada: {median_angle:.3f}")