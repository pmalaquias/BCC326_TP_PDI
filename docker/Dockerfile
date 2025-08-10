# Usa uma imagem base da NVIDIA que já tem CUDA e cuDNN instalados.
# Isso simplifica o processo para usar a GPU.
FROM tensorflow/tensorflow:latest-gpu-jupyter

# Define o diretório de trabalho dentro do container.
WORKDIR /app

# Instala o Python 3.10 e o gerenciador de pacotes pip.
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 1. Instala as dependências mais essenciais
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    "numpy<2.0" \
    matplotlib \
    opencv-python \
    pandas \
    scikit-learn \
    seaborn

# 2. Instala pacotes que podem causar dependências complexas
RUN pip install --no-cache-dir \
    dataclasses \
    tensorflow_hub

# Copia o restante dos arquivos do seu projeto para o container.
COPY . .

# Define o comando padrão para iniciar o Jupyter Notebook
CMD ["jupyter", "lab", "--port=8888", "--ip=0.0.0.0", "--allow-root"]