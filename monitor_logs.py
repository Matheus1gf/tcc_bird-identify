#!/usr/bin/env python3
"""
Monitor de logs em tempo real com filtros e cores
"""

import time
import os
from datetime import datetime

def monitor_logs(log_file="logs/debug.log", filter_level=None):
    """Monitora logs em tempo real com filtros opcionais"""
    
    if not os.path.exists(log_file):
        print(f"❌ Arquivo de log não encontrado: {log_file}")
        return
    
    print(f"🔍 Monitorando: {log_file}")
    print(f"📅 Iniciado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Cores para diferentes níveis
    colors = {
        'ERROR': '\033[91m',    # Vermelho
        'WARNING': '\033[93m',  # Amarelo
        'INFO': '\033[92m',     # Verde
        'DEBUG': '\033[94m',    # Azul
        'RESET': '\033[0m'      # Reset
    }
    
    try:
        with open(log_file, 'r') as f:
            # Vai para o final do arquivo
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # Aplica filtro se especificado
                    if filter_level and filter_level.upper() not in line:
                        continue
                    
                    # Aplica cores baseadas no nível
                    colored_line = line
                    for level, color in colors.items():
                        if level != 'RESET' and level in line:
                            colored_line = line.replace(level, f"{color}{level}{colors['RESET']}")
                    
                    print(colored_line.strip())
                else:
                    time.sleep(0.1)
                    
    except KeyboardInterrupt:
        print("\n🛑 Monitoramento interrompido pelo usuário")
    except Exception as e:
        print(f"❌ Erro no monitoramento: {e}")

if __name__ == "__main__":
    import sys
    
    # Verifica argumentos
    filter_level = None
    if len(sys.argv) > 1:
        filter_level = sys.argv[1]
    
    print("📊 Monitor de Logs em Tempo Real")
    print("Uso: python monitor_logs.py [ERROR|WARNING|INFO|DEBUG]")
    print()
    
    monitor_logs(filter_level=filter_level)
