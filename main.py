#!/usr/bin/env python3
"""
Sistema de Identificação de Pássaros com IA
TCC - 2025

Ponto de entrada principal do sistema
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from interfaces.web_app import main

if __name__ == "__main__":
    main()
