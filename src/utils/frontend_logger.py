#!/usr/bin/env python3
"""
Sistema de Logs Frontend para Console do Navegador
Integrado com Streamlit e sistema existente
"""

import streamlit as st
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum

class FrontendLogLevel(Enum):
    """Níveis de log para frontend"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warn"
    ERROR = "error"
    SUCCESS = "success"

@dataclass
class FrontendLogEntry:
    """Entrada de log para frontend"""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    extra_data: Dict[str, Any] = None
    
    def to_console_string(self) -> str:
        """Converte para string formatada para console do navegador"""
        return f"[{self.timestamp}] {self.level.upper()} [{self.module}.{self.function}] {self.message}"

class FrontendLogger:
    """Sistema de logs para frontend (console do navegador)"""
    
    def __init__(self):
        self.logs: List[FrontendLogEntry] = []
        self.max_logs = 1000
    
    def log(self, level: FrontendLogLevel, message: str, module: str = "", 
            function: str = "", extra_data: Dict[str, Any] = None):
        """Adiciona log"""
        log_entry = FrontendLogEntry(
            timestamp=datetime.now().strftime("%H:%M:%S.%f")[:-3],
            level=level.value,
            message=message,
            module=module,
            function=function,
            extra_data=extra_data or {}
        )
        
        self.logs.append(log_entry)
        
        # Manter apenas os últimos max_logs
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        # Enviar para console do navegador
        self._send_to_console(log_entry)
    
    def _send_to_console(self, log_entry: FrontendLogEntry):
        """Envia log para console do navegador via JavaScript"""
        console_script = f"""
        <script>
        console.{log_entry.level}('{log_entry.to_console_string()}');
        </script>
        """
        st.markdown(console_script, unsafe_allow_html=True)
    
    def debug(self, message: str, module: str = "", function: str = "", **kwargs):
        """Log de debug"""
        self.log(FrontendLogLevel.DEBUG, message, module, function, kwargs)
    
    def info(self, message: str, module: str = "", function: str = "", **kwargs):
        """Log de informação"""
        self.log(FrontendLogLevel.INFO, message, module, function, kwargs)
    
    def warning(self, message: str, module: str = "", function: str = "", **kwargs):
        """Log de aviso"""
        self.log(FrontendLogLevel.WARNING, message, module, function, kwargs)
    
    def error(self, message: str, module: str = "", function: str = "", **kwargs):
        """Log de erro"""
        self.log(FrontendLogLevel.ERROR, message, module, function, kwargs)
    
    def success(self, message: str, module: str = "", function: str = "", **kwargs):
        """Log de sucesso"""
        self.log(FrontendLogLevel.SUCCESS, message, module, function, kwargs)
    
    def get_logs(self, level: Optional[FrontendLogLevel] = None, limit: int = 100) -> List[FrontendLogEntry]:
        """Retorna logs filtrados"""
        logs = self.logs
        
        if level:
            logs = [log for log in logs if log.level == level.value]
        
        return logs[-limit:]
    
    def clear_logs(self):
        """Limpa todos os logs"""
        self.logs.clear()
    
    def render_logs_widget(self):
        """Renderiza widget de logs para interface"""
        st.markdown("""
        <div id="frontend-logs-widget">
            <h4><i class="bi bi-terminal"></i> Logs do Console</h4>
            <div id="logs-container" style="background: #f8f9fa; padding: 10px; border-radius: 5px; max-height: 300px; overflow-y: auto;">
                <div id="logs-content"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # JavaScript para atualizar logs em tempo real
        logs_script = """
        <script>
        function updateLogs() {
            const logsContainer = document.getElementById('logs-content');
            if (logsContainer) {
                // Simular logs em tempo real
                const now = new Date();
                const timeStr = now.toLocaleTimeString();
                const logEntry = document.createElement('div');
                logEntry.innerHTML = `<span style="color: #6c757d;">[${timeStr}]</span> <span style="color: #007bff;">INFO</span> <span style="color: #28a745;">[FrontendLogger.updateLogs]</span> Sistema funcionando normalmente`;
                logsContainer.appendChild(logEntry);
                logsContainer.scrollTop = logsContainer.scrollHeight;
            }
        }
        
        // DESABILITADO: Auto-refresh causa hot-reload infinito
        // Atualizar logs a cada 2 segundos
        // setInterval(updateLogs, 2000); // FIXED: Comentado para prevenir loop infinito
        </script>
        """
        st.markdown(logs_script, unsafe_allow_html=True)

# Instância global do logger frontend
frontend_logger = FrontendLogger()

# Funções de conveniência
def log_debug_frontend(message: str, module: str = "", function: str = "", **kwargs):
    """Log de debug frontend global"""
    frontend_logger.debug(message, module, function, **kwargs)

def log_info_frontend(message: str, module: str = "", function: str = "", **kwargs):
    """Log de informação frontend global"""
    frontend_logger.info(message, module, function, **kwargs)

def log_warning_frontend(message: str, module: str = "", function: str = "", **kwargs):
    """Log de aviso frontend global"""
    frontend_logger.warning(message, module, function, **kwargs)

def log_error_frontend(message: str, module: str = "", function: str = "", **kwargs):
    """Log de erro frontend global"""
    frontend_logger.error(message, module, function, **kwargs)

def log_success_frontend(message: str, module: str = "", function: str = "", **kwargs):
    """Log de sucesso frontend global"""
    frontend_logger.success(message, module, function, **kwargs)

def render_frontend_logs():
    """Renderiza logs frontend na interface"""
    frontend_logger.render_logs_widget()
