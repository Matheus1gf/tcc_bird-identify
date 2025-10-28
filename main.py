#!/usr/bin/env python3
"""
Sistema de Identificação de Pássaros com IA
TCC - 2025

Ponto de entrada principal do sistema
"""

# Suprimir warnings SSL/OpenSSL antes de qualquer importação
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")
warnings.filterwarnings("ignore", message=".*urllib3 v2 only supports OpenSSL 1.1.1+.*")
warnings.filterwarnings("ignore", message=".*NotOpenSSLWarning.*")
# Suprimir warnings do TensorFlow
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
warnings.filterwarnings("ignore", message=".*TensorFlow.*")
warnings.filterwarnings("ignore", message=".*keras.*")
# Suprimir warnings gerais
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interfaces.web_app import main

if __name__ == "__main__":
    main()
