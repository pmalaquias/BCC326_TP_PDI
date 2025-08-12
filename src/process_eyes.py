import cv2
import os
import numpy as np
from pathlib import Path

def detect_and_crop_eyes(image_path, output_path, cascade_path=None):
    """
    Detecta olhos em uma imagem usando Haar Cascade e salva a versão croppada.
    
    Args:
        image_path (str): Caminho para a imagem de entrada
        output_path (str): Caminho para salvar a imagem croppada
        cascade_path (str): Caminho para o arquivo Haar Cascade (opcional)
    """
    # Carrega a imagem
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        return False
    
    # Converte para escala de cinza para detecção
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Carrega o classificador Haar Cascade para olhos
    if cascade_path and os.path.exists(cascade_path):
        eye_cascade = cv2.CascadeClassifier(cascade_path)
    else:
        # Usa o classificador padrão do OpenCV
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    if eye_cascade.empty():
        print("Erro ao carregar o classificador Haar Cascade")
        return False
    
    # Detecta os olhos
    eyes = eye_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(eyes) == 0:
        print(f"Nenhum olho detectado em: {image_path}")
        return False
    
    # Para cada olho detectado, faz o crop e salva
    for i, (x, y, w, h) in enumerate(eyes):
        # Adiciona uma margem ao redor do olho para melhor visualização
        margin = 10
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(image.shape[1], x + w + margin)
        y_end = min(image.shape[0], y + h + margin)
        
        # Faz o crop do olho
        eye_crop = image[y_start:y_end, x_start:x_end]
        
        # Gera nome único para cada olho detectado
        base_name = Path(image_path).stem
        output_filename = f"{base_name}_eye_{i+1}.jpg"
        output_filepath = os.path.join(output_path, output_filename)
        
        # Salva a imagem croppada
        cv2.imwrite(output_filepath, eye_crop)
        print(f"Olho {i+1} salvo em: {output_filepath}")
    
    return True

def process_all_images(input_folder, output_folder, cascade_path=None):
    """
    Processa todas as imagens .jpg em uma pasta.
    
    Args:
        input_folder (str): Pasta com as imagens de entrada
        output_folder (str): Pasta para salvar as imagens croppadas
        cascade_path (str): Caminho para o arquivo Haar Cascade (opcional)
    """
    # Cria a pasta de saída se não existir
    os.makedirs(output_folder, exist_ok=True)
    
    # Lista todas as imagens .jpg na pasta de entrada
    image_extensions = ['.jpg', '.jpeg', '.JPG', '.JPEG']
    image_files = []
    
    for file in os.listdir(input_folder):
        if any(file.lower().endswith(ext.lower()) for ext in image_extensions):
            image_files.append(os.path.join(input_folder, file))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada em: {input_folder}")
        return
    
    print(f"Encontradas {len(image_files)} imagens para processar")
    
    # Processa cada imagem
    processed_count = 0
    for image_path in image_files:
        print(f"\nProcessando: {os.path.basename(image_path)}")
        if detect_and_crop_eyes(image_path, output_folder, cascade_path):
            processed_count += 1
    
    print(f"\nProcessamento concluído! {processed_count} imagens processadas com sucesso.")
    print(f"Imagens croppadas salvas em: {output_folder}")

def main():
    """Função principal"""
    # Configurações
    input_folder = "imgs/unit_eyes"
    output_folder = "imgs/unit_eyes/cropped"
    
    # Caminho opcional para um classificador Haar Cascade personalizado
    # cascade_path = "models/haarcascade_eye.xml"  # Descomente se tiver um modelo personalizado
    
    print("=== Processador de Imagens para Detecção de Olhos ===")
    print(f"Pasta de entrada: {input_folder}")
    print(f"Pasta de saída: {output_folder}")
    
    # Verifica se a pasta de entrada existe
    if not os.path.exists(input_folder):
        print(f"Erro: A pasta {input_folder} não existe!")
        return
    
    # Processa as imagens
    process_all_images(input_folder, output_folder)

if __name__ == "__main__":
    main()
