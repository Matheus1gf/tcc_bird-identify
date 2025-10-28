#!/usr/bin/env python3
"""
Sistema de Supressão de Warnings SSL/OpenSSL
Para resolver warnings de compatibilidade
"""

import warnings
import os
import sys

def suppress_ssl_warnings():
    """Suprime warnings de SSL/OpenSSL"""
    try:
        # Suprimir warning do urllib3
        from urllib3.exceptions import NotOpenSSLWarning
        warnings.filterwarnings('ignore', category=NotOpenSSLWarning)
        
        # Suprimir warnings de SSL em geral
        warnings.filterwarnings('ignore', message='.*SSL.*')
        warnings.filterwarnings('ignore', message='.*OpenSSL.*')
        warnings.filterwarnings('ignore', message='.*LibreSSL.*')
        
        # Suprimir warnings de urllib3
        warnings.filterwarnings('ignore', message='.*urllib3.*')
        
        return True
        
    except ImportError:
        # Se urllib3 não estiver disponível, apenas suprimir warnings gerais
        warnings.filterwarnings('ignore', message='.*SSL.*')
        warnings.filterwarnings('ignore', message='.*OpenSSL.*')
        warnings.filterwarnings('ignore', message='.*LibreSSL.*')
        return True
        
    except Exception as e:
        print(f"Erro ao suprimir warnings SSL: {e}")
        return False

def suppress_tensorflow_warnings():
    """Suprime warnings do TensorFlow"""
    try:
        # Suprimir warnings do TensorFlow
        warnings.filterwarnings('ignore', message='.*TensorFlow.*')
        warnings.filterwarnings('ignore', message='.*tensorflow.*')
        warnings.filterwarnings('ignore', message='.*keras.*')
        
        # Definir variáveis de ambiente para suprimir logs do TensorFlow
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
        
        return True
        
    except Exception as e:
        print(f"Erro ao suprimir warnings TensorFlow: {e}")
        return False

def suppress_all_warnings():
    """Suprime todos os warnings relevantes"""
    try:
        # Suprimir warnings SSL/OpenSSL
        suppress_ssl_warnings()
        
        # Suprimir warnings TensorFlow
        suppress_tensorflow_warnings()
        
        # Suprimir outros warnings comuns
        warnings.filterwarnings('ignore', message='.*deprecated.*')
        warnings.filterwarnings('ignore', message='.*future.*')
        warnings.filterwarnings('ignore', message='.*UserWarning.*')
        
        return True
        
    except Exception as e:
        print(f"Erro ao suprimir warnings: {e}")
        return False

def enable_warnings():
    """Reabilita warnings (para debug)"""
    try:
        warnings.resetwarnings()
        return True
    except Exception as e:
        print(f"Erro ao reabilitar warnings: {e}")
        return False

# Suprimir warnings automaticamente ao importar
suppress_all_warnings()

# Funções de conveniência
def quiet_mode():
    """Ativa modo silencioso"""
    suppress_all_warnings()
    
def debug_mode():
    """Ativa modo debug (mostra warnings)"""
    enable_warnings()
    
def test_warnings():
    """Testa se os warnings estão sendo suprimidos"""
    try:
        import urllib3
        print("✅ urllib3 importado sem warnings")
        
        import ssl
        print(f"✅ SSL version: {ssl.OPENSSL_VERSION}")
        
        return True
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    print("=== TESTE DO SISTEMA DE SUPRESSÃO DE WARNINGS ===")
    test_warnings()
