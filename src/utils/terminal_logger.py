#!/usr/bin/env python3
"""
Sistema de Logs em Tempo Real para Terminal
Integrado com Streamlit e sistema existente
"""

import logging
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import queue
import os

class LogLevel(Enum):
    """Níveis de log"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SUCCESS = "SUCCESS"

@dataclass
class TerminalLogEntry:
    """Entrada de log para terminal"""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    thread_id: str
    process_id: str
    extra_data: Dict[str, Any] = None
    
    def to_terminal_string(self) -> str:
        """Converte para string formatada para terminal"""
        # Cores para terminal
        colors = {
            "DEBUG": "\033[36m",      # Cyan
            "INFO": "\033[34m",       # Blue
            "WARNING": "\033[33m",    # Yellow
            "ERROR": "\033[31m",      # Red
            "CRITICAL": "\033[35m",   # Magenta
            "SUCCESS": "\033[32m"     # Green
        }
        
        reset = "\033[0m"
        color = colors.get(self.level, "")
        
        # Ícones para cada nível
        icons = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨",
            "SUCCESS": "✅"
        }
        
        icon = icons.get(self.level, "📝")
        
        # Formatar timestamp
        dt = datetime.fromisoformat(self.timestamp)
        time_str = dt.strftime("%H:%M:%S.%f")[:-3]
        
        # Formatar mensagem
        if self.module and self.function:
            context = f"[{self.module}.{self.function}]"
        elif self.module:
            context = f"[{self.module}]"
        else:
            context = ""
        
        return f"{color}{time_str} {icon} {self.level}{reset} {context} {self.message}"

class TerminalLogger:
    """Sistema de logs em tempo real para terminal"""
    
    def __init__(self, log_file: str = "logs/terminal.log", max_logs: int = 1000):
        self.log_file = log_file
        self.max_logs = max_logs
        self.logs: List[TerminalLogEntry] = []
        self.log_queue = queue.Queue()
        self.subscribers: List[callable] = []
        self.running = False
        self.log_thread = None
        
        # Criar diretório de logs se não existir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        # Configurar logging básico
        self._setup_basic_logging()
        
        # Iniciar thread de processamento
        self.start()
    
    def _setup_basic_logging(self):
        """Configura logging básico"""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def start(self):
        """Inicia o sistema de logs"""
        if not self.running:
            self.running = True
            self.log_thread = threading.Thread(target=self._process_logs, daemon=True)
            self.log_thread.start()
    
    def stop(self):
        """Para o sistema de logs"""
        self.running = False
        if self.log_thread:
            self.log_thread.join(timeout=1)
    
    def _process_logs(self):
        """Processa logs em tempo real"""
        while self.running:
            try:
                # Processar logs da fila
                while not self.log_queue.empty():
                    log_entry = self.log_queue.get_nowait()
                    self._add_log(log_entry)
                    self._notify_subscribers(log_entry)
                
                time.sleep(0.1)  # Pequena pausa para não sobrecarregar CPU
                
            except Exception as e:
                print(f"Erro no processamento de logs: {e}")
                time.sleep(1)
    
    def _add_log(self, log_entry: TerminalLogEntry):
        """Adiciona log à lista e exibe no terminal"""
        self.logs.append(log_entry)
        
        # Manter apenas os últimos max_logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Exibir no terminal
        print(log_entry.to_terminal_string())
        
        # Salvar no arquivo
        self._save_to_file(log_entry)
    
    def _save_to_file(self, log_entry: TerminalLogEntry):
        """Salva log no arquivo"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"{log_entry.timestamp} [{log_entry.level}] {log_entry.module}.{log_entry.function}: {log_entry.message}\n")
        except Exception as e:
            print(f"Erro ao salvar log: {e}")
    
    def _notify_subscribers(self, log_entry: TerminalLogEntry):
        """Notifica subscribers sobre novo log"""
        for subscriber in self.subscribers:
            try:
                subscriber(log_entry)
            except Exception as e:
                print(f"Erro ao notificar subscriber: {e}")
    
    def subscribe(self, callback: callable):
        """Inscreve callback para receber logs em tempo real"""
        self.subscribers.append(callback)
    
    def unsubscribe(self, callback: callable):
        """Remove callback dos subscribers"""
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def log(self, level: LogLevel, message: str, module: str = "", 
            function: str = "", line: int = 0, extra_data: Dict[str, Any] = None):
        """Adiciona log"""
        log_entry = TerminalLogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.value,
            message=message,
            module=module,
            function=function,
            line=line,
            thread_id=str(threading.get_ident()),
            process_id=str(os.getpid()),
            extra_data=extra_data or {}
        )
        
        self.log_queue.put(log_entry)
    
    def debug(self, message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
        """Log de debug"""
        self.log(LogLevel.DEBUG, message, module, function, line, kwargs)
    
    def info(self, message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
        """Log de informação"""
        self.log(LogLevel.INFO, message, module, function, line, kwargs)
    
    def warning(self, message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
        """Log de aviso"""
        self.log(LogLevel.WARNING, message, module, function, line, kwargs)
    
    def error(self, message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
        """Log de erro"""
        self.log(LogLevel.ERROR, message, module, function, line, kwargs)
    
    def critical(self, message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
        """Log crítico"""
        self.log(LogLevel.CRITICAL, message, module, function, line, kwargs)
    
    def success(self, message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
        """Log de sucesso"""
        self.log(LogLevel.SUCCESS, message, module, function, line, kwargs)
    
    def get_logs(self, level: Optional[LogLevel] = None, limit: int = 100) -> List[TerminalLogEntry]:
        """Retorna logs filtrados"""
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.level == level.value]
        
        return logs[-limit:]
    
    def clear_logs(self):
        """Limpa todos os logs"""
        self.logs.clear()

# Instância global do logger
terminal_logger = TerminalLogger()

# Funções de conveniência
def log_debug(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de debug global"""
    terminal_logger.debug(message, module, function, line, **kwargs)

def log_info(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de informação global"""
    terminal_logger.info(message, module, function, line, **kwargs)

def log_warning(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de aviso global"""
    terminal_logger.warning(message, module, function, line, **kwargs)

def log_error(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de erro global"""
    terminal_logger.error(message, module, function, line, **kwargs)

def log_critical(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log crítico global"""
    terminal_logger.critical(message, module, function, line, **kwargs)

def log_success(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de sucesso global"""
    terminal_logger.success(message, module, function, line, **kwargs)
