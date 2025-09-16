#!/usr/bin/env python3
"""
Módulo simples de debug para botões
"""

import logging
from datetime import datetime

# Configurar logger
logger = logging.getLogger('button_debug')

class ButtonDebugLogger:
    """Logger simples para debug de botões"""
    
    def __init__(self):
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = f"button_debug_{self.session_id}.log"
        
    def log_button_click(self, button_name, data=None):
        """Log de clique em botão"""
        message = f"[{datetime.now()}] CLICK: {button_name}"
        if data:
            message += f" | Data: {data}"
        logger.info(message)
        
    def log_step(self, step_name, data=None):
        """Log de passo"""
        message = f"[{datetime.now()}] STEP: {step_name}"
        if data:
            message += f" | Data: {data}"
        logger.info(message)
        
    def log_file_check(self, file_path, exists):
        """Log de verificação de arquivo"""
        status = "EXISTS" if exists else "NOT_FOUND"
        message = f"[{datetime.now()}] FILE_CHECK: {file_path} | Status: {status}"
        logger.info(message)
        
    def log_error(self, error_msg, error_type="UNKNOWN"):
        """Log de erro"""
        message = f"[{datetime.now()}] ERROR [{error_type}]: {error_msg}"
        logger.error(message)
        
    def log_success(self, success_msg):
        """Log de sucesso"""
        message = f"[{datetime.now()}] SUCCESS: {success_msg}"
        logger.info(message)
        
    def log_manual_analysis_call(self, file_path, data):
        """Log de chamada para análise manual"""
        message = f"[{datetime.now()}] MANUAL_ANALYSIS_CALL: {file_path}"
        logger.info(message)
        
    def log_manual_analysis_result(self, result_path, success):
        """Log de resultado da análise manual"""
        status = "SUCCESS" if success else "FAILED"
        message = f"[{datetime.now()}] MANUAL_ANALYSIS_RESULT: {result_path} | Status: {status}"
        logger.info(message)
        
    def log_file_cleanup(self, file_path, removed):
        """Log de limpeza de arquivo"""
        status = "REMOVED" if removed else "NOT_FOUND"
        message = f"[{datetime.now()}] FILE_CLEANUP: {file_path} | Status: {status}"
        logger.info(message)
        
    def log_session_end(self):
        """Log de fim de sessão"""
        message = f"[{datetime.now()}] SESSION_END: {self.session_id}"
        logger.info(message)

# Instância global
button_debug = ButtonDebugLogger()
