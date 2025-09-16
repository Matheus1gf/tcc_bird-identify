#!/usr/bin/env python3
"""
Script de Inicialização Automática do Sistema
Inicia sincronização contínua e aplicação Streamlit
"""

import os
import sys
import time
import threading
import subprocess
from pathlib import Path

def start_sync_system():
    """Inicia sistema de sincronização em background"""
    try:
        from learning_sync import start_continuous_sync
        start_continuous_sync()
        print("✅ Sistema de sincronização iniciado")
        return True
    except Exception as e:
        print(f"❌ Erro ao iniciar sincronização: {e}")
        return False

def start_streamlit_app():
    """Inicia aplicação Streamlit"""
    try:
        # Determinar comando baseado no SO
        if sys.platform == "win32":
            cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        else:
            cmd = ["python3", "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
        
        print("🚀 Iniciando aplicação Streamlit...")
        print(f"📱 Acesse: http://localhost:8501")
        
        # Executar aplicação
        subprocess.run(cmd)
        
    except KeyboardInterrupt:
        print("\n⏹️ Aplicação interrompida pelo usuário")
    except Exception as e:
        print(f"❌ Erro ao iniciar aplicação: {e}")

def main():
    """Função principal"""
    print("🎯 INICIANDO SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 60)
    
    # Verificar se estamos no diretório correto
    if not os.path.exists("app.py"):
        print("❌ Arquivo app.py não encontrado!")
        print("   Execute este script no diretório do projeto")
        sys.exit(1)
    
    # Iniciar sincronização em background
    print("🔄 Iniciando sistema de sincronização...")
    sync_success = start_sync_system()
    
    if not sync_success:
        print("⚠️ Continuando sem sincronização...")
    
    # Aguardar um pouco para sincronização inicializar
    time.sleep(2)
    
    # Iniciar aplicação Streamlit
    print("\n🚀 Iniciando aplicação principal...")
    start_streamlit_app()

if __name__ == "__main__":
    main()
