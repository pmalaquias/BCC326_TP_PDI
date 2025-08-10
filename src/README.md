# Código Python - Conversão dos Notebooks

Esta pasta contém todos os arquivos Python convertidos dos notebooks Jupyter (.ipynb) do projeto.

## Arquivos Disponíveis

### 1. `detector_olhar.py`

- **Origem**: Notebook original
- **Função**: Detecção de olhar usando classificadores Haar e modelo de gaze
- **Características**:
  - Carrega classificadores Haar para rosto e olhos
  - Utiliza modelo de gaze treinado
  - Processa imagens para detecção de olhar

### 2. `processamento_teste_IA.py`

- **Origem**: Notebook original
- **Função**: Teste e validação de modelos de IA
- **Características**:
  - Carrega dados do arquivo CSV
  - Testa modelos treinados
  - Avalia performance dos modelos

### 3. `finetune.py`

- **Origem**: `notebooks/finetune.ipynb`
- **Função**: Fine-tuning do modelo de atenção usando dados reais do MPIIGaze
- **Características**:
  - Verifica estrutura do dataset MPIIGaze
  - Processa dados reais para ajuste fino
  - Configura caminhos e verifica GPU
  - Preparado para treinamento com dados reais

### 4. `pre_processamento.py`

- **Origem**: `notebooks/pre_processamento.ipynb`
- **Função**: Pré-processamento dos dados do UnityEyes para criar dataset de atenção
- **Características**:
  - Processa arquivos JSON do UnityEyes
  - Extrai vetores de olhar e tamanho da pupila
  - Cria rótulos binários de atenção
  - Salva dataset em CSV

### 5. `processamento.py`

- **Origem**: `notebooks/processamento.ipynb`
- **Função**: Processamento e treinamento de LSTM para análise de olhar
- **Características**:
  - Carrega dados processados do CSV
  - Explora e visualiza dados
  - Pré-processa dados para LSTM
  - Cria, treina e avalia modelo LSTM
  - Salva modelo treinado

### 6. `tp_pdi.py`

- **Origem**: `notebooks/tp_pdi.ipynb`
- **Função**: Trabalho prático de PDI - Análise de Gaze e Detecção de Atenção
- **Características**:
  - Processa dataset MPIIGaze
  - Cria anotações de landmarks
  - Cria modelo CNN para classificação de gaze
  - Salva modelo treinado

### 7. `download_mpiigaze.py`

- **Origem**: `notebooks/Untitled-1.ipynb`
- **Função**: Download e organização do dataset MPIIGaze
- **Características**:
  - Copia arquivos do cache do Kaggle
  - Organiza estrutura de pastas
  - Verifica integridade dos dados

## Como Executar

Todos os arquivos podem ser executados diretamente do terminal:

```bash
# Navegue para a pasta src
cd src

# Execute qualquer arquivo
python detector_olhar.py
python processamento.py
python finetune.py
# etc.
```

## Dependências

Certifique-se de ter as seguintes bibliotecas instaladas:

```bash
pip install opencv-python
pip install tensorflow
pip install numpy
pip install pandas
pip install matplotlib
pip install seaborn
pip install scikit-learn
pip install h5py
pip install scipy
```

## Estrutura de Pastas

Os arquivos Python assumem a seguinte estrutura de pastas:

```
BCC326_TP_PDI/
├── src/                    # Esta pasta
├── classifier/            # Classificadores Haar
├── models/               # Modelos treinados
├── output/               # Arquivos de saída (CSV, etc.)
├── UnityEyes_Windows/    # Dados do UnityEyes
└── mpiigaze_real/        # Dataset MPIIGaze (se baixado)
```

## Observações

- Todos os caminhos foram atualizados para refletir a nova estrutura de pastas
- Os arquivos mantêm a funcionalidade original dos notebooks
- Código foi organizado em funções para melhor modularização
- Adicionados comentários e documentação para facilitar manutenção
