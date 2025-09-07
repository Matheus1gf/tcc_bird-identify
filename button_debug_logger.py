#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sistema de Logging Específico para Debug do Botão "Marcar para Análise Manual"
"""

import logging
import os
from datetime import datetime

class ButtonDebugLogger:
    """Logger específico para debug do botão de análise manual"""
    
    def __init__(self, log_file="button_debug.log"):
        self.log_file = log_file
        self.setup_logger()
    
    def setup_logger(self):
        """Configura o logger específico para o botão"""
        # Criar logger específico
        self.logger = logging.getLogger('button_debug')
        self.logger.setLevel(logging.DEBUG)
        
        # Limpar handlers existentes
        self.logger.handlers.clear()
        
        # Criar handler para arquivo
        file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Criar formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Adicionar handler ao logger
        self.logger.addHandler(file_handler)
        
        # Log inicial
        self.logger.info("=" * 80)
        self.logger.info("INICIANDO LOG DE DEBUG DO BOTÃO 'MARCAR PARA ANÁLISE MANUAL'")
        self.logger.info("=" * 80)
    
    def log_button_click(self, button_key, temp_path=None):
        """Log quando o botão é clicado"""
        self.logger.info(f"🖱️ BOTÃO CLICADO: {button_key}")
        self.logger.info(f"📁 Arquivo temporário esperado: {temp_path}")
        self.logger.info(f"⏰ Timestamp: {datetime.now().isoformat()}")
    
    def log_file_check(self, temp_path, exists):
        """Log da verificação de arquivo temporário"""
        self.logger.info(f"📂 VERIFICAÇÃO DE ARQUIVO TEMPORÁRIO:")
        self.logger.info(f"   Caminho: {temp_path}")
        self.logger.info(f"   Existe: {exists}")
        if exists:
            try:
                file_size = os.path.getsize(temp_path)
                self.logger.info(f"   Tamanho: {file_size} bytes")
            except Exception as e:
                self.logger.error(f"   Erro ao verificar tamanho: {e}")
    
    def log_manual_analysis_call(self, temp_path, detection_data):
        """Log da chamada para análise manual"""
        self.logger.info(f"🔄 CHAMANDO MANUAL_ANALYSIS.ADD_IMAGE_FOR_ANALYSIS:")
        self.logger.info(f"   Arquivo: {temp_path}")
        self.logger.info(f"   Dados: {detection_data}")
    
    def log_manual_analysis_result(self, pending_path, success):
        """Log do resultado da análise manual"""
        self.logger.info(f"📋 RESULTADO DA ANÁLISE MANUAL:")
        self.logger.info(f"   Caminho pendente: {pending_path}")
        self.logger.info(f"   Sucesso: {success}")
        if success:
            try:
                file_size = os.path.getsize(pending_path)
                self.logger.info(f"   Tamanho do arquivo salvo: {file_size} bytes")
            except Exception as e:
                self.logger.error(f"   Erro ao verificar arquivo salvo: {e}")
    
    def log_file_cleanup(self, temp_path, removed):
        """Log da limpeza do arquivo temporário"""
        self.logger.info(f"🗑️ LIMPEZA DE ARQUIVO TEMPORÁRIO:")
        self.logger.info(f"   Arquivo: {temp_path}")
        self.logger.info(f"   Removido: {removed}")
    
    def log_error(self, error_msg, error_type="UNKNOWN"):
        """Log de erro"""
        self.logger.error(f"❌ ERRO [{error_type}]: {error_msg}")
    
    def log_success(self, success_msg):
        """Log de sucesso"""
        self.logger.info(f"✅ SUCESSO: {success_msg}")
    
    def log_step(self, step_name, details=""):
        """Log de passo genérico"""
        self.logger.info(f"🔍 PASSO: {step_name}")
        if details:
            self.logger.info(f"   Detalhes: {details}")
    
    def log_session_end(self):
        """Log do fim da sessão"""
        self.logger.info("=" * 80)
        self.logger.info("FIM DA SESSÃO DE DEBUG DO BOTÃO")
        self.logger.info("=" * 80)
        self.logger.info("")

# Instância global do logger
button_debug = ButtonDebugLogger()
