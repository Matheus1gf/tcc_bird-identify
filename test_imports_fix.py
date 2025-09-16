#!/usr/bin/env python3
"""
Teste específico para verificar se os imports estão funcionando
"""

import sys
import os

# Adicionar src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_core_imports():
    """Testar imports dos módulos core"""
    print("🔍 Testando imports dos módulos core...")
    
    try:
        print("  📦 Testando core.intuition...")
        from core.intuition import IntuitionEngine
        print("  ✅ core.intuition OK")
        
        print("  📦 Testando core.annotator...")
        from core.annotator import GradCAMAnnotator
        print("  ✅ core.annotator OK")
        
        print("  📦 Testando core.reasoning...")
        from core.reasoning import LogicalAIReasoningSystem
        print("  ✅ core.reasoning OK")
        
        print("  📦 Testando core.learning...")
        from core.learning import ContinuousLearningSystem
        print("  ✅ core.learning OK")
        
        print("  📦 Testando core.cache...")
        from core.cache import image_cache
        print("  ✅ core.cache OK")
        
        print("  📦 Testando core.learning_sync...")
        from core.learning_sync import stop_continuous_sync
        print("  ✅ core.learning_sync OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro nos imports core: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interfaces_imports():
    """Testar imports das interfaces"""
    print("\n🔍 Testando imports das interfaces...")
    
    try:
        print("  📦 Testando interfaces.manual_analysis...")
        from interfaces.manual_analysis import manual_analysis
        print("  ✅ interfaces.manual_analysis OK")
        
        print("  📦 Testando interfaces.tinder_interface...")
        from interfaces.tinder_interface import TinderInterface
        print("  ✅ interfaces.tinder_interface OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro nos imports interfaces: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_utils_imports():
    """Testar imports dos utils"""
    print("\n🔍 Testando imports dos utils...")
    
    try:
        print("  📦 Testando utils.debug_logger...")
        from utils.debug_logger import debug_logger
        print("  ✅ utils.debug_logger OK")
        
        print("  📦 Testando utils.button_debug...")
        from utils.button_debug import button_debug
        print("  ✅ utils.button_debug OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro nos imports utils: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_web_app_import():
    """Testar import do web_app"""
    print("\n🔍 Testando import do web_app...")
    
    try:
        print("  📦 Testando interfaces.web_app...")
        from interfaces.web_app import main
        print("  ✅ interfaces.web_app OK")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro no import web_app: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_module_initialization():
    """Testar inicialização dos módulos"""
    print("\n🔍 Testando inicialização dos módulos...")
    
    try:
        # Importar as classes primeiro
        from core.intuition import IntuitionEngine
        from core.reasoning import LogicalAIReasoningSystem
        from core.learning import ContinuousLearningSystem
        from interfaces.tinder_interface import TinderInterface
        from interfaces.manual_analysis import manual_analysis
        
        print("  🧠 Testando IntuitionEngine...")
        intuition_engine = IntuitionEngine("yolov8n.pt", "modelo_classificacao_passaros.keras")
        print("  ✅ IntuitionEngine inicializado")
        
        print("  🤖 Testando LogicalAIReasoningSystem...")
        reasoning_system = LogicalAIReasoningSystem()
        print("  ✅ LogicalAIReasoningSystem inicializado")
        
        print("  📚 Testando ContinuousLearningSystem...")
        learning_system = ContinuousLearningSystem("yolov8n.pt", "modelo_classificacao_passaros.keras")
        print("  ✅ ContinuousLearningSystem inicializado")
        
        print("  💡 Testando TinderInterface...")
        tinder_interface = TinderInterface(manual_analysis)
        print("  ✅ TinderInterface inicializado")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Erro na inicialização: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Função principal"""
    print("🧪 TESTE ESPECÍFICO DE IMPORTS")
    print("=" * 50)
    
    tests = [
        ("Imports Core", test_core_imports),
        ("Imports Interfaces", test_interfaces_imports),
        ("Imports Utils", test_utils_imports),
        ("Import Web App", test_web_app_import),
        ("Inicialização Módulos", test_module_initialization)
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
    
    # Resumo final
    print("\n" + "=" * 50)
    print("📊 RESUMO DOS TESTES")
    print("=" * 50)
    
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
    success = main()
    sys.exit(0 if success else 1)
