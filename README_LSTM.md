# Treinamento de LSTM para Análise de Olhar

Este projeto implementa um modelo LSTM (Long Short-Term Memory) para análise de olhar usando dados do dataset UnityEyes.

## Estrutura do Projeto

```
├── processamento.py          # Script principal de treinamento LSTM
├── gaze_labels.csv          # Dados processados (6.186 registros)
├── pre_processamento.ipynb  # Notebook de pré-processamento original
├── requirements.txt         # Dependências do projeto
├── install_dependencies.py  # Script de instalação automática
├── README_LSTM.md          # Este arquivo
└── UnityEyes_Windows/      # Dataset original
```

## Características dos Dados

### Dados de Entrada
- **Vetores de olhar**: `look_vec_x`, `look_vec_y`, `look_vec_z` (coordenadas 3D)
- **Tamanho da pupila**: `pupil_size` (valor normalizado)
- **Sequências temporais**: 10 timesteps por sequência

### Variável Target
- **Attention**: Classificação binária (0 = Sem atenção, 1 = Com atenção)
- **Distribuição**: 91.6% sem atenção, 8.4% com atenção

## Arquitetura da LSTM

O modelo implementa uma arquitetura LSTM com:

1. **3 camadas LSTM**:
   - LSTM 1: 128 unidades (return_sequences=True)
   - LSTM 2: 64 unidades (return_sequences=True)
   - LSTM 3: 32 unidades (return_sequences=False)

2. **Camadas de regularização**:
   - BatchNormalization após cada LSTM
   - Dropout (0.3) após cada camada

3. **Camadas densas**:
   - Dense 1: 64 unidades (ReLU)
   - Dense 2: 32 unidades (ReLU)
   - Dense 3: 2 unidades (Softmax) - saída

## Instalação e Configuração

### Pré-requisitos

- Python 3.6 ou superior
- pip (gerenciador de pacotes Python)

### Método 1: Instalação Automática (Recomendado)

1. **Clone ou baixe o projeto**:
```bash
git clone <url-do-repositorio>
cd BCC326_TP_PDI
```

2. **Execute o script de instalação automática**:
```bash
python3 install_dependencies.py
```

Este script tentará instalar as versões mais recentes das dependências e, se falhar, tentará versões mais antigas automaticamente.

### Método 2: Instalação Manual

1. **Crie um ambiente virtual (recomendado)**:
```bash
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

2. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

### Método 3: Instalação Individual

Se os métodos acima falharem, tente instalar cada pacote individualmente:

```bash
# Versões mais conservadoras
pip install pandas>=0.20.0
pip install numpy>=1.14.0
pip install matplotlib>=2.0.0
pip install seaborn>=0.8.0
pip install scikit-learn>=0.19.0
pip install tensorflow>=1.15.0
```

### Solução de Problemas de Instalação

#### Erro de versão do pandas
```bash
# Tente versões mais antigas
pip install pandas==0.24.2
# ou
pip install pandas==0.20.3
```

#### Erro de versão do TensorFlow
```bash
# Para Python 3.6-3.8
pip install tensorflow==1.15.0
# Para Python 3.8+
pip install tensorflow==2.0.0
```

#### Erro de dependências conflitantes
```bash
# Limpe o cache do pip
pip cache purge
# Reinstale
pip install --no-cache-dir -r requirements.txt
```

## Como Executar

### Execução Simples

```bash
python3 processamento.py
```

### Execução com Ambiente Virtual

```bash
# Ative o ambiente virtual primeiro
source venv/bin/activate  # Linux/Mac
python processamento.py
```

## Blocos do Código

### 1. Importação de Bibliotecas
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn para pré-processamento
- TensorFlow/Keras para LSTM

### 2. Carregamento e Exploração dos Dados
- Carrega `gaze_labels.csv`
- Visualiza distribuições e correlações
- Analisa valores nulos e estatísticas

### 3. Pré-processamento dos Dados
- Normalização com StandardScaler
- Criação de sequências temporais (10 timesteps)
- Divisão treino/validação/teste (60%/20%/20%)
- Codificação one-hot para target

### 4. Definição do Modelo LSTM
- Arquitetura sequencial com 3 camadas LSTM
- Regularização com BatchNormalization e Dropout
- Otimizador Adam com learning rate 0.001

### 5. Treinamento do Modelo
- Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
- Máximo 100 épocas
- Batch size 32
- Monitoramento de val_loss e val_accuracy

### 6. Avaliação do Modelo
- Métricas: acurácia, precisão, recall, F1-score
- Matriz de confusão
- Gráficos de treinamento (acurácia e loss)

## Saídas do Sistema

1. **Modelo salvo**: `melhor_modelo_lstm.h5`
2. **Visualizações**:
   - Distribuição das variáveis
   - Matriz de correlação
   - Curvas de treinamento
   - Matriz de confusão

3. **Métricas de performance**:
   - Acurácia no conjunto de teste
   - Relatório de classificação detalhado

## Configurações Ajustáveis

- `sequence_length`: Comprimento da sequência (padrão: 10)
- `batch_size`: Tamanho do batch (padrão: 32)
- `epochs`: Número máximo de épocas (padrão: 100)
- `learning_rate`: Taxa de aprendizado (padrão: 0.001)

## Análise de Resultados

O modelo LSTM é especialmente adequado para este problema porque:

1. **Sequências temporais**: Os dados de olhar formam sequências naturais
2. **Dependências temporais**: A atenção pode depender de padrões anteriores
3. **Memória de longo prazo**: LSTM pode capturar padrões complexos ao longo do tempo

## Possíveis Melhorias

1. **Balanceamento de classes**: Técnicas como SMOTE ou class weights
2. **Hiperparâmetros**: Otimização com GridSearch ou RandomSearch
3. **Arquitetura**: Experimentar com Bidirectional LSTM ou GRU
4. **Features**: Adicionar features derivadas (velocidade, aceleração)
5. **Ensemble**: Combinar múltiplos modelos

## Troubleshooting

### Problemas Comuns

1. **Memória insuficiente**: Reduzir batch_size ou sequence_length
2. **Overfitting**: Aumentar dropout ou reduzir complexidade do modelo
3. **Underfitting**: Aumentar complexidade ou reduzir regularização
4. **Dados desbalanceados**: Usar class_weights ou técnicas de balanceamento

### Erros de Instalação

1. **TensorFlow não encontrado**:
   ```bash
   pip install tensorflow
   ```

2. **Pandas não encontrado**:
   ```bash
   pip install pandas
   ```

3. **Problemas de versão do Python**:
   - Certifique-se de usar Python 3.6+
   - Use ambiente virtual para evitar conflitos

### Logs e Debugging

O código inclui logs detalhados em cada etapa:
- Shape dos dados
- Progresso do treinamento
- Métricas de performance
- Visualizações automáticas

## Exemplo de Execução

```bash
# 1. Instalar dependências
python3 install_dependencies.py

# 2. Executar o script
python3 processamento.py

# 3. Saída esperada:
# === TREINAMENTO DE LSTM PARA ANÁLISE DE OLHAR ===
# Dataset: UnityEyes - Dados processados
# 
# Carregando dados...
# Dataset carregado com 6186 registros
# Colunas disponíveis: ['filename', 'look_vec_x', 'look_vec_y', 'look_vec_z', 'pupil_size', 'attention']
# 
# === EXPLORAÇÃO DOS DADOS ===
# ...
``` 