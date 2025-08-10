#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pré-processamento dos dados do UnityEyes para criar dataset de atenção
Baseado no notebook pre_processamento.ipynb
"""

import os
import json
import numpy as np
import pandas as pd

def processar_dados_unityeyes():
    """
    Processa os dados do UnityEyes para criar o dataset de atenção
    """
    # 1. Defina o caminho para a pasta com as imagens e JSONs
    data_dir = '../UnityEyes_Windows/UnityEyes_Windows/imgs'

    # Verifique se o diretório existe
    if not os.path.exists(data_dir):
        print(f"Erro: O diretório '{data_dir}' não foi encontrado. Verifique o caminho.")
        return None
    else:
        print(f"Diretório '{data_dir}' encontrado.")

        # 2. Inicialize uma lista para armazenar os dados de todos os frames
        all_data = []

        # 3. Percorra todos os arquivos na pasta
        for filename in os.listdir(data_dir):
            # Encontre apenas os arquivos JSON
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)

                try:
                    # Carregue o arquivo JSON
                    with open(file_path, 'r') as f:
                        data = json.load(f)

                    # 4. Extraia os dados relevantes
                    look_vec_str = data['eye_details']['look_vec']
                    # Converta a string "(x, y, z, w)" para uma tupla de floats
                    look_vec_coords = tuple(map(float, look_vec_str.strip('()').split(',')))
                    look_vec_x, look_vec_y, look_vec_z = look_vec_coords[0:3]

                    # Tente extrair o pupil_size, se existir
                    try:
                        pupil_size = float(data['eye_details']['pupil_size'])
                    except (KeyError, ValueError):
                        pupil_size = np.nan # Use NaN se não for encontrado

                    # 5. Crie a lógica para o rótulo de atenção
                    # Vamos assumir que a atenção está no centro da tela, com um vetor de olhar ideal (0, 0, -1)
                    # Calcule a distância Euclidiana entre o vetor de olhar e o vetor de atenção
                    distance_to_center = np.sqrt(look_vec_x**2 + look_vec_y**2 + (look_vec_z - (-1))**2)

                    # Defina um limiar para a atenção. Você pode ajustar este valor.
                    attention_threshold = 0.2

                    # Atribua o rótulo binário
                    attention_label = 1 if distance_to_center < attention_threshold else 0

                    # 6. Armazene os dados em um dicionário
                    all_data.append({
                        'filename': filename,
                        'look_vec_x': look_vec_x,
                        'look_vec_y': look_vec_y,
                        'look_vec_z': look_vec_z,
                        'pupil_size': pupil_size,
                        'attention': attention_label
                    })

                except (IOError, json.JSONDecodeError, KeyError) as e:
                    print(f"Erro ao processar o arquivo {filename}: {e}")
                    continue # Pula para o próximo arquivo em caso de erro

        # 7. Converta a lista de dicionários para um DataFrame do pandas
        gaze_df = pd.DataFrame(all_data)

        # 8. Salve o DataFrame em um arquivo CSV na pasta output
        output_path = os.path.join('..', 'output', 'gaze_labels.csv')
        gaze_df.to_csv(output_path, index=False)
        print(f"\nArquivo '{output_path}' criado com sucesso! Contém {len(gaze_df)} registros.")

        # Exiba as primeiras linhas do novo DataFrame para verificar
        print("\nPrimeiras linhas do novo CSV:")
        print(gaze_df.head())
        
        print("\nDistribuição dos rótulos de atenção:")
        print(gaze_df['attention'].value_counts())
        
        return gaze_df

def main():
    """
    Função principal
    """
    print("=== PRÉ-PROCESSAMENTO DOS DADOS UNITYEYES ===")
    
    # Processa os dados do UnityEyes
    gaze_df = processar_dados_unityeyes()
    
    if gaze_df is not None:
        print(f"\nDataset criado com sucesso!")
        print(f"Total de registros: {len(gaze_df)}")
        print(f"Registros com atenção: {len(gaze_df[gaze_df['attention'] == 1])}")
        print(f"Registros sem atenção: {len(gaze_df[gaze_df['attention'] == 0])}")
    else:
        print("Erro ao processar os dados do UnityEyes.")

if __name__ == "__main__":
    main()
