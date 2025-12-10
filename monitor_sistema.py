#!/usr/bin/env python3
"""
Monitor de Sistema - Acompanhamento em Tempo Real
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path

def clear_screen():
    """Limpa a tela do terminal"""
    os.system('clear' if os.name != 'nt' else 'cls')

def get_file_size_mb(filepath):
    """Retorna tamanho do arquivo em MB"""
    try:
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / (1024 * 1024)
            return f"{size:.2f} MB"
    except:
        pass
    return "N/A"

def get_json_count(filepath):
    """Conta items em um JSON"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return len(data)
                elif isinstance(data, list):
                    return len(data)
    except:
        pass
    return 0

def monitor_sistema():
    """Monitora o sistema em tempo real"""
    
    clear_screen()
    print("=" * 80)
    print("🔍 MONITOR DE SISTEMA - TCC Bird Identify")
    print("=" * 80)
    print()
    
    # Verificar processos
    print("📊 PROCESSOS:")
    os.system("ps aux | grep streamlit | grep -v grep || echo '  ❌ Streamlit não está rodando'")
    print()
    
    # Verificar portas
    print("🌐 PORTAS:")
    os.system("lsof -i :8501 2>/dev/null || echo '  ❌ Porta 8501 não está em uso'")
    print()
    
    # Status dos arquivos de dados
    print("💾 ARQUIVOS DE DADOS:")
    
    data_files = {
        'Episodic Memory': 'data/episodic_memory.json',
        'Causal Reasoning': 'data/causal_reasoning.json',
        'Abstract Inference': 'data/abstract_inference.json',
        'Analogical Reasoning': 'data/analogical_reasoning.json',
        'Meta Learning': 'data/meta_learning.json',
        'Few Shot Learning': 'data/few_shot_learning.json',
    }
    
    for name, filepath in data_files.items():
        size = get_file_size_mb(filepath)
        count = get_json_count(filepath)
        status = "✅" if os.path.exists(filepath) else "❌"
        print(f"  {status} {name:25} {filepath:30} (Size: {size}, Items: {count})")
    
    print()
    
    # Status dos logs
    print("📝 LOGS:")
    log_files = {
        'Debug Log': 'logs/debug.log',
        'Realtime Log': 'logs/realtime.log',
        'Terminal Log': 'logs/terminal.log',
    }
    
    for name, filepath in log_files.items():
        size = get_file_size_mb(filepath)
        status = "✅" if os.path.exists(filepath) else "❌"
        print(f"  {status} {name:25} {filepath:30} (Size: {size})")
    
    print()
    
    # Últimas linhas do log de debug
    print("📊 ÚLTIMAS 10 LINHAS DO DEBUG LOG:")
    print("-" * 80)
    try:
        os.system("tail -n 10 logs/debug.log 2>/dev/null || echo '  ❌ Log não encontrado'")
    except:
        print("  ❌ Erro ao ler log")
    
    print()
    print("-" * 80)
    print(f"⏰ Última atualização: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    print("Pressione Ctrl+C para sair")
    print()

def main():
    """Função principal"""
    import sys
    import signal
    
    def signal_handler(sig, frame):
        print("\n\n🛑 Monitoramento encerrado pelo usuário")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while True:
            clear_screen()
            monitor_sistema()
            time.sleep(5)  # Atualiza a cada 5 segundos
    except KeyboardInterrupt:
        print("\n\n🛑 Monitoramento encerrado")
    except Exception as e:
        print(f"\n❌ Erro no monitoramento: {e}")

if __name__ == "__main__":
    main()

