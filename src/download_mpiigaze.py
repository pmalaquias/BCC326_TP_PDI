#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download e Organização do Dataset MPIIGaze
Baseado no notebook Untitled-1.ipynb
"""

import os
import shutil

def baixar_mpiigaze():
    """
    Baixa e organiza o dataset MPIIGaze
    """
    print("=== DOWNLOAD E ORGANIZAÇÃO DO MPIIGAZE ===")
    
    # 1. Defina o caminho do cache do Kaggle
    # (Este caminho é o que o seu código de download retornou)
    path_cache_kaggle = r"C:\Users\pedro\.cache\kagglehub\datasets\dhruv413\mpiigaze\versions\1"

    # 2. Defina a pasta de destino na raiz do seu projeto
    path_destino = "../mpiigaze_real"

    # Crie a pasta de destino se ela não existir
    if not os.path.exists(path_destino):
        os.makedirs(path_destino)
        print(f"Pasta de destino criada: {path_destino}")

    # 3. Copie os arquivos da pasta de cache para a pasta do seu projeto
    # O 'shutil.copytree' é usado para copiar todo o diretório de dados
    # do MPIIGaze que está dentro do cache.
    source_dir = os.path.join(path_cache_kaggle, 'MPIIGaze')
    destination_dir = os.path.join(path_destino, 'MPIIGaze')

    if os.path.exists(source_dir):
        # Remove o diretório de destino se já existir
        if os.path.exists(destination_dir):
            shutil.rmtree(destination_dir)
            print("Diretório de destino existente removido.")
        
        # Copia a estrutura de diretórios do cache para o projeto
        shutil.copytree(source_dir, destination_dir)
        print("Arquivos do MPIIGaze copiados para a raiz do projeto com sucesso!")
        print(f"Origem: {source_dir}")
        print(f"Destino: {destination_dir}")
    else:
        print(f"Erro: O diretório de origem '{source_dir}' não foi encontrado.")
        print("Verifique se o dataset foi baixado corretamente do Kaggle.")
        return False

    # 4. (Opcional) Limpe o arquivo zip baixado se estiver na raiz
    if os.path.exists('../mpiigaze.zip'):
        os.remove('../mpiigaze.zip')
        print("Arquivo zip removido.")

    return True

def verificar_estrutura():
    """
    Verifica se a estrutura do dataset foi criada corretamente
    """
    print("\n=== VERIFICAÇÃO DA ESTRUTURA ===")
    
    path_destino = "../mpiigaze_real"
    destination_dir = os.path.join(path_destino, 'MPIIGaze')
    
    if os.path.exists(destination_dir):
        print(f"✅ Dataset MPIIGaze encontrado em: {destination_dir}")
        
        # Lista o conteúdo da pasta principal
        try:
            contents = os.listdir(destination_dir)
            print(f"Conteúdo da pasta MPIIGaze:")
            for item in contents:
                item_path = os.path.join(destination_dir, item)
                if os.path.isdir(item_path):
                    print(f"  📁 {item}/")
                else:
                    print(f"  📄 {item}")
        except Exception as e:
            print(f"Erro ao listar conteúdo: {e}")
    else:
        print(f"❌ Dataset MPIIGaze não encontrado em: {destination_dir}")

def main():
    """
    Função principal
    """
    print("=== DOWNLOAD E ORGANIZAÇÃO DO DATASET MPIIGAZE ===")
    
    # Baixa e organiza o dataset
    sucesso = baixar_mpiigaze()
    
    if sucesso:
        # Verifica a estrutura criada
        verificar_estrutura()
        print("\n✅ Processo concluído com sucesso!")
    else:
        print("\n❌ Erro no processo de download/organização.")

if __name__ == "__main__":
    main()
