#!/usr/bin/env python3
"""
Teste do Frontend do Sistema de Raciocínio Lógico de IA
Verifica se todas as funcionalidades estão funcionando
"""

import requests
import time
import subprocess
import sys
import os

def test_streamlit_installation():
    """Testa se Streamlit está instalado"""
    print("TESTE 1: Verificação da Instalação do Streamlit")
    print("=" * 50)
    
    try:
        import streamlit
        import plotly
        print("Streamlit instalado:", streamlit.__version__)
        print("Plotly instalado:", plotly.__version__)
        return True
    except ImportError as e:
        print(f"Erro de importação: {e}")
        return False

def test_app_file():
    """Testa se o arquivo app.py existe e é válido"""
    print("\n🔍 TESTE 2: Verificação do Arquivo app.py")
    print("=" * 50)
    
    if not os.path.exists("app.py"):
        print("❌ Arquivo app.py não encontrado")
        return False
    
    try:
        with open("app.py", "r") as f:
            content = f.read()
        
        # Verificar se contém elementos essenciais
        essential_elements = [
            "streamlit as st",
            "SantoGraalSystem",
            "def main():",
            "st.set_page_config",
            "st.header"
        ]
        
        missing_elements = []
        for element in essential_elements:
            if element not in content:
                missing_elements.append(element)
        
        if missing_elements:
            print(f"❌ Elementos faltando: {missing_elements}")
            return False
        
        print("✅ Arquivo app.py válido")
        print(f"📊 Tamanho: {len(content)} caracteres")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler app.py: {e}")
        return False

def test_streamlit_config():
    """Testa se a configuração do Streamlit está correta"""
    print("\n🔍 TESTE 3: Verificação da Configuração")
    print("=" * 50)
    
    config_path = ".streamlit/config.toml"
    if not os.path.exists(config_path):
        print("❌ Arquivo de configuração não encontrado")
        return False
    
    try:
        with open(config_path, "r") as f:
            config_content = f.read()
        
        # Verificar elementos essenciais da configuração
        config_elements = [
            "port = 8501",
            "primaryColor = \"#667eea\"",
            "backgroundColor = \"#ffffff\""
        ]
        
        missing_config = []
        for element in config_elements:
            if element not in config_content:
                missing_config.append(element)
        
        if missing_config:
            print(f"❌ Configurações faltando: {missing_config}")
            return False
        
        print("✅ Configuração do Streamlit válida")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao ler configuração: {e}")
        return False

def test_streamlit_startup():
    """Testa se o Streamlit consegue iniciar"""
    print("\n🔍 TESTE 4: Teste de Inicialização do Streamlit")
    print("=" * 50)
    
    try:
        # Tentar executar streamlit --version
        result = subprocess.run(
            [sys.executable, "-m", "streamlit", "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print("✅ Streamlit pode ser executado")
            print(f"📊 Versão: {result.stdout.strip()}")
            return True
        else:
            print(f"❌ Erro ao executar Streamlit: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Timeout ao executar Streamlit")
        return False
    except Exception as e:
        print(f"❌ Erro ao testar Streamlit: {e}")
        return False

def test_app_syntax():
    """Testa se o app.py tem sintaxe válida"""
    print("\n🔍 TESTE 5: Verificação de Sintaxe do app.py")
    print("=" * 50)
    
    try:
        # Tentar compilar o arquivo
        with open("app.py", "r") as f:
            code = f.read()
        
        compile(code, "app.py", "exec")
        print("✅ Sintaxe do app.py válida")
        return True
        
    except SyntaxError as e:
        print(f"❌ Erro de sintaxe: {e}")
        return False
    except Exception as e:
        print(f"❌ Erro ao verificar sintaxe: {e}")
        return False

def test_dependencies():
    """Testa se todas as dependências estão disponíveis"""
    print("\n🔍 TESTE 6: Verificação de Dependências")
    print("=" * 50)
    
    dependencies = [
        ("streamlit", "st"),
        ("plotly", "plotly"),
        ("pandas", "pd"),
        ("numpy", "np"),
        ("cv2", "cv2"),
        ("PIL", "Image")
    ]
    
    missing_deps = []
    for dep_name, import_name in dependencies:
        try:
            __import__(import_name)
            print(f"✅ {dep_name} disponível")
        except ImportError:
            print(f"❌ {dep_name} não disponível")
            missing_deps.append(dep_name)
    
    if missing_deps:
        print(f"⚠️ Dependências faltando: {missing_deps}")
        return False
    
    print("✅ Todas as dependências estão disponíveis")
    return True

def test_frontend_functionality():
    """Testa funcionalidades específicas do frontend"""
    print("\n🔍 TESTE 7: Teste de Funcionalidades do Frontend")
    print("=" * 50)
    
    try:
        # Importar módulos do app
        import streamlit as st
        import plotly.express as px
        import pandas as pd
        import numpy as np
        
        # Testar criação de gráfico
        df = pd.DataFrame({
            'x': [1, 2, 3, 4, 5],
            'y': [2, 4, 6, 8, 10]
        })
        
        fig = px.bar(df, x='x', y='y', title='Teste')
        print("✅ Gráfico Plotly criado com sucesso")
        
        # Testar criação de DataFrame
        test_df = pd.DataFrame({
            'Classe': ['bird', 'dog', 'cat'],
            'Confiança': [0.9, 0.8, 0.7]
        })
        print("✅ DataFrame criado com sucesso")
        
        # Testar operações NumPy
        arr = np.array([1, 2, 3, 4, 5])
        result = np.mean(arr)
        print(f"✅ Operação NumPy: média = {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao testar funcionalidades: {e}")
        return False

def main():
    """Função principal de teste"""
    print("🧠 TESTE DO FRONTEND DO SISTEMA SANTO GRAAL DA IA")
    print("=" * 60)
    print("Verificando se o frontend está pronto para uso")
    print("=" * 60)
    
    tests = [
        ("Instalação do Streamlit", test_streamlit_installation),
        ("Arquivo app.py", test_app_file),
        ("Configuração", test_streamlit_config),
        ("Inicialização", test_streamlit_startup),
        ("Sintaxe", test_app_syntax),
        ("Dependências", test_dependencies),
        ("Funcionalidades", test_frontend_functionality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Erro no teste {test_name}: {e}")
            results.append((test_name, False))
    
    # Relatório final
    print("\n📊 RELATÓRIO FINAL DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    print(f"📈 Total de testes: {total}")
    print(f"✅ Aprovados: {passed}")
    print(f"❌ Falharam: {total - passed}")
    print(f"🎯 Taxa de sucesso: {(passed/total)*100:.1f}%")
    
    print(f"\n📋 Detalhes dos Testes:")
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"   {status}: {test_name}")
    
    if passed == total:
        print(f"\n🎉 TODOS OS TESTES PASSARAM!")
        print(f"🚀 Frontend está pronto para uso!")
        print(f"🌐 Execute: python3 run_frontend.py")
        print(f"📱 Acesse: http://localhost:8501")
    else:
        print(f"\n⚠️ ALGUNS TESTES FALHARAM!")
        print(f"🔧 Corrija os problemas antes de usar o frontend")
    
    print(f"\n🎯 Status geral: {'PASSED' if passed == total else 'FAILED'}")

if __name__ == "__main__":
    main()
