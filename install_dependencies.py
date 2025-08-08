#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para instalar dependências com fallback para versões mais antigas
"""

import subprocess
import sys

def install_package(package_name, versions):
    """
    Tenta instalar um pacote com diferentes versões
    """
    for version in versions:
        try:
            print(f"Tentando instalar {package_name}{version}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package_name}{version}"])
            print(f"✓ {package_name}{version} instalado com sucesso!")
            return True
        except subprocess.CalledProcessError:
            print(f"✗ Falha ao instalar {package_name}{version}")
            continue
    return False

def main():
    print("=== INSTALAÇÃO DE DEPENDÊNCIAS ===")
    print("Tentando instalar as bibliotecas necessárias...\n")
    
    # Lista de pacotes com versões alternativas (da mais recente para a mais antiga)
    packages = {
        'pandas': ['>=1.3.0', '>=1.0.0', '>=0.24.0', '>=0.20.0', ''],
        'numpy': ['>=1.21.0', '>=1.16.0', '>=1.14.0', ''],
        'matplotlib': ['>=3.5.0', '>=3.0.0', '>=2.0.0', ''],
        'seaborn': ['>=0.11.0', '>=0.9.0', '>=0.8.0', ''],
        'scikit-learn': ['>=1.0.0', '>=0.20.0', '>=0.19.0', ''],
        'tensorflow': ['>=2.8.0', '>=2.0.0', '>=1.15.0', '']
    }
    
    failed_packages = []
    
    for package, versions in packages.items():
        if not install_package(package, versions):
            failed_packages.append(package)
            print(f"⚠️  Não foi possível instalar {package}")
        print()
    
    if failed_packages:
        print("=== PACOTES QUE FALHARAM ===")
        for package in failed_packages:
            print(f"- {package}")
        print("\nTente instalar manualmente:")
        for package in failed_packages:
            print(f"pip install {package}")
    else:
        print("=== TODAS AS DEPENDÊNCIAS INSTALADAS COM SUCESSO! ===")
        print("Agora você pode executar: python3 processamento.py")

if __name__ == "__main__":
    main() 