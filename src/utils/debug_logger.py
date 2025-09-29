#!/usr/bin/env python3
"""
Sistema de Debug Logger
Sistema de logging para debug da aplicação
"""

import logging
import os
from datetime import datetime

class DebugLogger:
    def __init__(self, name="debug_logger"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Criar diretório de logs se não existir
        log_dir = "logs"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Configurar handler para arquivo
        log_file = os.path.join(log_dir, "debug.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Configurar handler para console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formato das mensagens
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Adicionar handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def debug(self, message):
        """Log de debug"""
        self.logger.debug(f"🐛 {message}")
    
    def info(self, message):
        """Log de informação"""
        self.logger.info(f"[INFO] {message}")
    
    def warning(self, message):
        """Log de aviso"""
        self.logger.warning(f"[ALERTA] {message}")
    
    def error(self, message):
        """Log de erro"""
        self.logger.error(f"[ERRO] {message}")
    
    def critical(self, message):
        """Log crítico"""
        self.logger.critical(f"🚨 {message}")
    
    def success(self, message):
        """Log de sucesso"""
        self.logger.info(f"[SUCESSO] {message}")
    
    def button_click(self, button_name, details=""):
        """Log específico para cliques de botão"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"🔘 Botão '{button_name}' clicado às {timestamp}"
        if details:
            message += f" - {details}"
        self.logger.info(message)
    
    def image_upload(self, filename, size=None):
        """Log específico para upload de imagem"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[CAMERA] Imagem '{filename}' carregada às {timestamp}"
        if size:
            message += f" - Tamanho: {size}"
        self.logger.info(message)
    
    def analysis_start(self, analysis_type):
        """Log de início de análise"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logger.info(f"[BUSCA] Iniciando análise '{analysis_type}' às {timestamp}")
    
    def analysis_complete(self, analysis_type, result=None):
        """Log de conclusão de análise"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[SUCESSO] Análise '{analysis_type}' concluída às {timestamp}"
        if result:
            message += f" - Resultado: {result}"
        self.logger.info(message)
    
    def error_handling(self, error, context=""):
        """Log específico para tratamento de erros"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        message = f"[ERRO] Erro às {timestamp}"
        if context:
            message += f" - Contexto: {context}"
        message += f" - Erro: {str(error)}"
        self.logger.error(message)
    
    def log_session_start(self, filename):
        """Log de início de sessão"""
        self.logger.info(f"[RAPIDO] Sessão iniciada - Arquivo: {filename}")
    
    def log_error(self, message, error_type=""):
        """Log de erro com tipo"""
        if error_type:
            self.logger.error(f"[ERRO] [{error_type}] {message}")
        else:
            self.logger.error(f"[ERRO] {message}")
    
    def log_success(self, message):
        """Log de sucesso"""
        self.logger.info(f"[SUCESSO] {message}")

# Instância global do logger
debug_logger = DebugLogger()

# Funções de conveniência
def log_debug(message):
    """Função de conveniência para debug"""
    debug_logger.debug(message)

def log_info(message):
    """Função de conveniência para info"""
    debug_logger.info(message)

def log_warning(message):
    """Função de conveniência para warning"""
    debug_logger.warning(message)

def log_error(message):
    """Função de conveniência para error"""
    debug_logger.error(message)

def log_success(message):
    """Função de conveniência para success"""
    debug_logger.success(message)

def log_button_click(button_name, details=""):
    """Função de conveniência para clique de botão"""
    debug_logger.button_click(button_name, details)

def log_image_upload(filename, size=None):
    """Função de conveniência para upload de imagem"""
    debug_logger.image_upload(filename, size)

def log_analysis_start(analysis_type):
    """Função de conveniência para início de análise"""
    debug_logger.analysis_start(analysis_type)

def log_analysis_complete(analysis_type, result=None):
    """Função de conveniência para conclusão de análise"""
    debug_logger.analysis_complete(analysis_type, result)

def log_error_handling(error, context=""):
    """Função de conveniência para tratamento de erros"""
    debug_logger.error_handling(error, context)
