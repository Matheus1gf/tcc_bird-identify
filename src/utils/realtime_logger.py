#!/usr/bin/env python3
"""
Sistema de Logs em Tempo Real - Estilo CloudWatch AWS
"""

import logging
import json
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
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

@dataclass
class LogEntry:
    """Entrada de log estruturada"""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line: int
    thread_id: str
    process_id: str
    extra_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário"""
        return asdict(self)

class RealtimeLogger:
    """Sistema de logs em tempo real estilo CloudWatch"""
    
    def __init__(self, log_file: str = "logs/realtime.log", max_logs: int = 1000):
        self.log_file = log_file
        self.max_logs = max_logs
        self.logs: List[LogEntry] = []
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
                logging.StreamHandler()
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
    
    def _add_log(self, log_entry: LogEntry):
        """Adiciona log à lista"""
        self.logs.append(log_entry)
        
        # Manter apenas os últimos max_logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Salvar no arquivo
        self._save_to_file(log_entry)
    
    def _save_to_file(self, log_entry: LogEntry):
        """Salva log no arquivo"""
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry.to_dict(), ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"Erro ao salvar log: {e}")
    
    def _notify_subscribers(self, log_entry: LogEntry):
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
        log_entry = LogEntry(
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
    
    def get_logs(self, level: Optional[LogLevel] = None, limit: int = 100) -> List[LogEntry]:
        """Retorna logs filtrados"""
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.level == level.value]
        
        return logs[-limit:]
    
    def get_logs_by_module(self, module: str, limit: int = 100) -> List[LogEntry]:
        """Retorna logs por módulo"""
        return [log for log in self.logs if module in log.module][-limit:]
    
    def get_logs_by_time_range(self, start_time: str, end_time: str) -> List[LogEntry]:
        """Retorna logs por intervalo de tempo"""
        start_dt = datetime.fromisoformat(start_time)
        end_dt = datetime.fromisoformat(end_time)
        
        return [
            log for log in self.logs 
            if start_dt <= datetime.fromisoformat(log.timestamp) <= end_dt
        ]
    
    def clear_logs(self):
        """Limpa todos os logs"""
        self.logs.clear()
    
    def export_logs(self, file_path: str):
        """Exporta logs para arquivo JSON"""
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([log.to_dict() for log in self.logs], f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Erro ao exportar logs: {e}")

# Instância global do logger
realtime_logger = RealtimeLogger()

# Funções de conveniência
def log_debug(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de debug global"""
    realtime_logger.debug(message, module, function, line, **kwargs)

def log_info(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de informação global"""
    realtime_logger.info(message, module, function, line, **kwargs)

def log_warning(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de aviso global"""
    realtime_logger.warning(message, module, function, line, **kwargs)

def log_error(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de erro global"""
    realtime_logger.error(message, module, function, line, **kwargs)

def log_success(message: str, module: str = "", function: str = "", line: int = 0, **kwargs):
    """Log de sucesso global"""
    realtime_logger.info(f"SUCCESS: {message}", module, function, line, **kwargs)
