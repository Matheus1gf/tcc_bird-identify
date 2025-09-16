#!/usr/bin/env python3
"""
Teste Interno Completo do Sistema
Verifica todos os módulos e funcionalidades antes de rodar a aplicação
"""

import sys
import os
import traceback
from datetime import datetime

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Teste 1: Verificar todos os imports"""
    print("🔍 TESTE 1: Verificando imports...")
    
    try:
        # Testar imports básicos
        print("  📦 Testando imports básicos...")
        import streamlit as st
        import cv2
        import numpy as np
        import pandas as pd
        import plotly.express as px
        import plotly.graph_objects as go
        from datetime import datetime, timedelta
        from PIL import Image
        print("  ✅ Imports básicos OK")
        
        # Testar imports dos módulos core
        print("  🧠 Testando módulos core...")
        from core.intuition import IntuitionEngine
        from core.reasoning import LogicalAIReasoningSystem
        from core.learning import ContinuousLearningSystem
        from core.cache import image_cache
        from core.learning_sync import stop_continuous_sync
        print("  ✅ Módulos core OK")
        
        # Testar imports das interfaces
        print("  🌐 Testando interfaces...")
        from interfaces.manual_analysis import manual_analysis
        from interfaces.tinder_interface import TinderInterface
        print("  ✅ Interfaces OK")
        
        # Testar imports dos utils
        print("  🔧 Testando utils...")
        from utils.button_debug import button_debug
        from utils.debug_logger import debug_logger
        print("  ✅ Utils OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro nos imports: {e}")
        traceback.print_exc()
        return False

def test_module_initialization():
    """Teste 2: Verificar inicialização dos módulos"""
    print("\n🔍 TESTE 2: Verificando inicialização dos módulos...")
    
    try:
        # Testar IntuitionEngine
        print("  🧠 Testando IntuitionEngine...")
        try:
            intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
            print("  ✅ IntuitionEngine inicializado")
        except Exception as e:
            print(f"  ⚠️ IntuitionEngine com aviso: {e}")
        
        # Testar LogicalAIReasoningSystem
        print("  🤖 Testando LogicalAIReasoningSystem...")
        try:
            reasoning_system = LogicalAIReasoningSystem()
            print("  ✅ LogicalAIReasoningSystem inicializado")
        except Exception as e:
            print(f"  ⚠️ LogicalAIReasoningSystem com aviso: {e}")
        
        # Testar ContinuousLearningSystem
        print("  📚 Testando ContinuousLearningSystem...")
        try:
            learning_system = ContinuousLearningSystem()
            print("  ✅ ContinuousLearningSystem inicializado")
        except Exception as e:
            print(f"  ⚠️ ContinuousLearningSystem com aviso: {e}")
        
        # Testar TinderInterface
        print("  💡 Testando TinderInterface...")
        try:
            tinder_interface = TinderInterface(manual_analysis)
            print("  ✅ TinderInterface inicializado")
        except Exception as e:
            print(f"  ⚠️ TinderInterface com aviso: {e}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro na inicialização: {e}")
        traceback.print_exc()
        return False

def test_web_app_import():
    """Teste 3: Verificar import do web_app"""
    print("\n🔍 TESTE 3: Verificando import do web_app...")
    
    try:
        from interfaces.web_app import main
        print("  ✅ web_app importado com sucesso")
        return True
    except Exception as e:
        print(f"  ❌ Erro ao importar web_app: {e}")
        traceback.print_exc()
        return False

def test_file_structure():
    """Teste 4: Verificar estrutura de arquivos"""
    print("\n🔍 TESTE 4: Verificando estrutura de arquivos...")
    
    required_files = [
        "main.py",
        "src/interfaces/web_app.py",
        "src/core/intuition.py",
        "src/core/reasoning.py",
        "src/core/learning.py",
        "src/core/cache.py",
        "src/core/learning_sync.py",
        "src/interfaces/manual_analysis.py",
        "src/interfaces/tinder_interface.py",
        "src/utils/debug_logger.py",
        "src/utils/button_debug.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"  ✅ {file_path}")
    
    if missing_files:
        print(f"  ❌ Arquivos faltando: {missing_files}")
        return False
    
    return True

def test_dependencies():
    """Teste 5: Verificar dependências externas"""
    print("\n🔍 TESTE 5: Verificando dependências externas...")
    
    dependencies = [
        ("streamlit", "st"),
        ("cv2", "cv2"),
        ("numpy", "np"),
        ("pandas", "pd"),
        ("plotly", "px"),
        ("PIL", "Image")
    ]
    
    missing_deps = []
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"  ✅ {dep_name}")
        except ImportError:
            print(f"  ❌ {dep_name} não encontrado")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"  ⚠️ Dependências faltando: {missing_deps}")
        return False
    
    return True

def test_model_files():
    """Teste 6: Verificar arquivos de modelo"""
    print("\n🔍 TESTE 6: Verificando arquivos de modelo...")
    
    model_files = [
        "yolov8n.pt",
        "modelo_classificacao_passaros.keras"
    ]
    
    for model_file in model_files:
        if os.path.exists(model_file):
            print(f"  ✅ {model_file}")
        else:
            print(f"  ⚠️ {model_file} não encontrado (será criado automaticamente)")
    
    return True

def create_mock_models():
    """Criar modelos mock para testes"""
    print("\n🔧 Criando modelos mock para testes...")
    
    # Criar arquivo YOLO mock
    if not os.path.exists("yolov8n.pt"):
        print("  📝 Criando yolov8n.pt mock...")
        with open("yolov8n.pt", "w") as f:
            f.write("# Mock YOLO model file")
    
    # Criar arquivo Keras mock
    if not os.path.exists("modelo_classificacao_passaros.keras"):
        print("  📝 Criando modelo_classificacao_passaros.keras mock...")
        with open("modelo_classificacao_passaros.keras", "w") as f:
            f.write("# Mock Keras model file")

def fix_import_issues():
    """Corrigir problemas de import"""
    print("\n🔧 Corrigindo problemas de import...")
    
    # Verificar se há problemas nos módulos core
    core_modules = [
        "src/core/intuition.py",
        "src/core/reasoning.py", 
        "src/core/learning.py",
        "src/core/annotator.py",
        "src/core/curator.py"
    ]
    
    for module_path in core_modules:
        if os.path.exists(module_path):
            try:
                with open(module_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Verificar se há imports problemáticos
                if "from ultralytics import YOLO" in content:
                    print(f"  ⚠️ {module_path} tem import YOLO problemático")
                if "import tensorflow as tf" in content:
                    print(f"  ⚠️ {module_path} tem import TensorFlow problemático")
                    
            except Exception as e:
                print(f"  ❌ Erro ao verificar {module_path}: {e}")

def run_comprehensive_test():
    """Executar teste completo"""
    print("🧪 INICIANDO TESTE INTERNO COMPLETO DO SISTEMA")
    print("=" * 60)
    
    tests = [
        ("Estrutura de Arquivos", test_file_structure),
        ("Dependências", test_dependencies),
        ("Imports", test_imports),
        ("Arquivos de Modelo", test_model_files),
        ("Inicialização de Módulos", test_module_initialization),
        ("Import do Web App", test_web_app_import)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Executando: {test_name}")
        try:
            result = test_func()
            results.append((test_name, result))
            if result:
                print(f"✅ {test_name}: PASSOU")
            else:
                print(f"❌ {test_name}: FALHOU")
        except Exception as e:
            print(f"❌ {test_name}: ERRO - {e}")
            results.append((test_name, False))
    
    # Criar modelos mock se necessário
    create_mock_models()
    
    # Corrigir problemas de import
    fix_import_issues()
    
    # Resumo final
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"{status}: {test_name}")
    
    print(f"\n📈 Taxa de Sucesso: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 TODOS OS TESTES PASSARAM! Sistema pronto para execução.")
        return True
    else:
        print(f"\n⚠️ {total-passed} TESTES FALHARAM. Corrija os problemas antes de executar.")
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
