# Processador de Imagens para Detecção de Olhos

Este script Python detecta olhos em imagens usando o modelo Haar Cascade e salva versões croppadas contendo apenas a região do olho.

## Funcionalidades

- Detecta automaticamente olhos em imagens .jpg
- Faz crop da região do olho com margem adicional
- Processa todas as imagens em uma pasta
- Salva as imagens croppadas com nomes únicos
- Usa o classificador Haar Cascade padrão do OpenCV

## Requisitos

Instale as dependências necessárias:

```bash
pip install -r requirements.txt
```

## Como usar

1. **Execução simples:**

   ```bash
   python src/process_eyes.py
   ```

2. **Execução com classificador personalizado:**
   - Descomente a linha `cascade_path` no script
   - Coloque seu arquivo .xml do Haar Cascade na pasta `models/`

## Estrutura de pastas

```
imgs/
├── unit_eyes/           # Imagens originais
│   ├── 1.jpg
│   ├── 10.jpg
│   ├── 10001.jpg
│   └── ...
└── unit_eyes/cropped/   # Imagens croppadas (criada automaticamente)
    ├── 1_eye_1.jpg
    ├── 10_eye_1.jpg
    └── ...
```

## Configurações

O script está configurado para:

- **Pasta de entrada:** `imgs/unit_eyes`
- **Pasta de saída:** `imgs/unit_eyes/cropped`
- **Extensões suportadas:** .jpg, .jpeg, .JPG, .JPEG
- **Margem do crop:** 10 pixels ao redor do olho detectado

## Parâmetros de detecção

- `scaleFactor`: 1.1 (reduz a imagem em 10% a cada escala)
- `minNeighbors`: 5 (mínimo de vizinhos para confirmar detecção)
- `minSize`: (30, 30) (tamanho mínimo do olho em pixels)

## Saída

Para cada olho detectado, o script gera uma imagem separada com o nome:
`{nome_original}_eye_{numero}.jpg`

## Tratamento de erros

- Verifica se as imagens podem ser carregadas
- Cria automaticamente a pasta de saída
- Reporta imagens sem olhos detectados
- Continua processando mesmo se uma imagem falhar

## Personalização

Para usar um classificador Haar Cascade personalizado:

1. Coloque o arquivo .xml na pasta `models/`
2. Descomente a linha `cascade_path` no script
3. Ajuste o caminho conforme necessário
