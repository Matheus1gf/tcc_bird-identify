#!/bin/bash

echo "========================================"
echo "Instalação do Sistema IA Neuro-Simbólica"
echo "Identificação de Pássaros - macOS"
echo "========================================"
echo

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "ERRO: Python3 não encontrado!"
    echo "Por favor, instale Python 3.8+ usando:"
    echo "1. brew install python3"
    echo "2. Ou baixe de https://python.org"
    exit 1
fi

echo "Python encontrado!"
python3 --version
echo

# Verificar se pip está instalado
if ! command -v pip3 &> /dev/null; then
    echo "Instalando pip..."
    python3 -m ensurepip --upgrade
fi

# Atualizar pip
echo "Atualizando pip..."
python3 -m pip install --upgrade pip
echo

# Instalar dependências básicas
echo "Instalando dependências básicas..."
pip3 install numpy==1.24.3
pip3 install opencv-python==4.8.1.78
pip3 install tensorflow==2.13.0
pip3 install ultralytics==8.0.196
pip3 install matplotlib==3.7.2
pip3 install scikit-learn==1.3.0
pip3 install scikit-image==0.21.0
echo

# Instalar dependências para novos módulos
echo "Instalando dependências para módulos avançados..."
pip3 install networkx==3.1
pip3 install requests==2.31.0
pip3 install pillow==10.0.0
pip3 install pandas==2.0.3
echo

# Instalar dependências para APIs externas (opcional)
echo "Instalando dependências para APIs externas (opcional)..."
pip3 install google-generativeai==0.3.0
pip3 install openai==0.28.0
echo

# Instalar dependências para desenvolvimento
echo "Instalando dependências para desenvolvimento..."
pip3 install jupyter==1.0.0
pip3 install ipykernel==6.25.0
pip3 install notebook==7.0.0
echo

# Verificar instalação
echo "Verificando instalação..."
python3 -c "import cv2, numpy, tensorflow, ultralytics, networkx, matplotlib; print('Todas as dependências principais foram instaladas com sucesso!')"
if [ $? -ne 0 ]; then
    echo "ERRO: Falha na verificação das dependências"
    exit 1
fi

echo
echo "========================================"
echo "Instalação concluída com sucesso!"
echo "========================================"
echo
echo "Para testar o sistema:"
echo "1. Execute: python3 main.py"
echo "2. Ou execute: python3 enhanced_main.py"
echo
echo "Para configurar APIs externas:"
echo "1. Edite enhanced_main.py"
echo "2. Configure API_KEY_GEMINI ou API_KEY_GPT4V"
echo
echo "Para usar Jupyter Notebook:"
echo "1. Execute: jupyter notebook"
echo "2. Abra o arquivo desejado"
echo
