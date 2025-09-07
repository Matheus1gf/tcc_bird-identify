@echo off
echo ========================================
echo Instalacao do Sistema IA Neuro-Simbolica
echo Identificacao de Passaros - Windows
echo ========================================
echo.

REM Verificar se Python esta instalado
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERRO: Python nao encontrado!
    echo Por favor, instale Python 3.8+ de https://python.org
    echo Certifique-se de marcar "Add Python to PATH" durante a instalacao
    pause
    exit /b 1
)

echo Python encontrado!
python --version
echo.

REM Atualizar pip
echo Atualizando pip...
python -m pip install --upgrade pip
echo.

REM Instalar dependencias basicas
echo Instalando dependencias basicas...
pip install numpy==1.24.3
pip install opencv-python==4.8.1.78
pip install tensorflow==2.13.0
pip install ultralytics==8.0.196
pip install matplotlib==3.7.2
pip install scikit-learn==1.3.0
pip install scikit-image==0.21.0
echo.

REM Instalar dependencias para novos modulos
echo Instalando dependencias para modulos avancados...
pip install networkx==3.1
pip install requests==2.31.0
pip install pillow==10.0.0
pip install pandas==2.0.3
echo.

REM Instalar dependencias para APIs externas (opcional)
echo Instalando dependencias para APIs externas (opcional)...
pip install google-generativeai==0.3.0
pip install openai==0.28.0
echo.

REM Instalar dependencias para desenvolvimento
echo Instalando dependencias para desenvolvimento...
pip install jupyter==1.0.0
pip install ipykernel==6.25.0
pip install notebook==7.0.0
echo.

REM Verificar instalacao
echo Verificando instalacao...
python -c "import cv2, numpy, tensorflow, ultralytics, networkx, matplotlib; print('Todas as dependencias principais foram instaladas com sucesso!')"
if %errorlevel% neq 0 (
    echo ERRO: Falha na verificacao das dependencias
    pause
    exit /b 1
)

echo.
echo ========================================
echo Instalacao concluida com sucesso!
echo ========================================
echo.
echo Para testar o sistema:
echo 1. Execute: python main.py
echo 2. Ou execute: python enhanced_main.py
echo.
echo Para configurar APIs externas:
echo 1. Edite enhanced_main.py
echo 2. Configure API_KEY_GEMINI ou API_KEY_GPT4V
echo.
pause
