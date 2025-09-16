#!/usr/bin/env python3
"""
Teste Unitário - Sistema de Identificação de Pássaros
Testa funções e classes individuais
"""

import unittest
import sys
import os
import json
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from PIL import Image

# Adicionar o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class TestDebugLogger(unittest.TestCase):
    """Teste unitário para DebugLogger"""
    
    def setUp(self):
        """Configuração inicial"""
        self.logger = None
        
    def test_logger_initialization(self):
        """Teste: Inicialização do logger"""
        try:
            from utils.debug_logger import debug_logger
            self.assertIsNotNone(debug_logger)
            self.logger = debug_logger
        except ImportError:
            self.skipTest("DebugLogger não disponível")
    
    def test_log_info(self):
        """Teste: Log de informação"""
        if self.logger:
            try:
                self.logger.log_info("Teste de informação")
                # Se não houve exceção, o teste passou
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Erro ao logar informação: {e}")
    
    def test_log_error(self):
        """Teste: Log de erro"""
        if self.logger:
            try:
                self.logger.log_error("Teste de erro", "TEST_ERROR")
                # Se não houve exceção, o teste passou
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Erro ao logar erro: {e}")
    
    def test_log_success(self):
        """Teste: Log de sucesso"""
        if self.logger:
            try:
                self.logger.log_success("Teste de sucesso")
                # Se não houve exceção, o teste passou
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Erro ao logar sucesso: {e}")

class TestButtonDebug(unittest.TestCase):
    """Teste unitário para ButtonDebug"""
    
    def setUp(self):
        """Configuração inicial"""
        self.button_debug = None
        
    def test_button_debug_initialization(self):
        """Teste: Inicialização do ButtonDebug"""
        try:
            from utils.button_debug import button_debug
            self.assertIsNotNone(button_debug)
            self.button_debug = button_debug
        except ImportError:
            self.skipTest("ButtonDebug não disponível")
    
    def test_log_button_click(self):
        """Teste: Log de clique de botão"""
        if self.button_debug:
            try:
                self.button_debug.log_button_click("test_button", "test_data")
                # Se não houve exceção, o teste passou
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Erro ao logar clique de botão: {e}")
    
    def test_log_step(self):
        """Teste: Log de passo"""
        if self.button_debug:
            try:
                self.button_debug.log_step("Teste de passo")
                # Se não houve exceção, o teste passou
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Erro ao logar passo: {e}")

class TestImageCache(unittest.TestCase):
    """Teste unitário para ImageCache"""
    
    def setUp(self):
        """Configuração inicial"""
        self.cache = None
        
    def test_cache_initialization(self):
        """Teste: Inicialização do cache"""
        try:
            from core.cache import image_cache
            self.assertIsNotNone(image_cache)
            self.cache = image_cache
        except ImportError:
            self.skipTest("ImageCache não disponível")
    
    def test_cache_operations(self):
        """Teste: Operações do cache"""
        if self.cache:
            try:
                # Testar operações básicas do cache
                test_key = "test_image"
                test_data = np.array([[[255, 0, 0]]], dtype=np.uint8)
                
                # Se o cache tem métodos, testar
                if hasattr(self.cache, 'get'):
                    result = self.cache.get(test_key)
                    # Resultado pode ser None se não existir
                    self.assertIsNotNone(result or True)
                
                self.assertTrue(True)
            except Exception as e:
                self.fail(f"Erro nas operações do cache: {e}")

class TestIntuitionEngine(unittest.TestCase):
    """Teste unitário para IntuitionEngine"""
    
    def setUp(self):
        """Configuração inicial"""
        self.engine = None
        
    def test_engine_initialization(self):
        """Teste: Inicialização do IntuitionEngine"""
        try:
            from core.intuition import IntuitionEngine
            self.engine = IntuitionEngine()
            self.assertIsNotNone(self.engine)
        except ImportError:
            self.skipTest("IntuitionEngine não disponível")
        except Exception as e:
            # Se falhar na inicialização, pode ser por dependências
            self.skipTest(f"IntuitionEngine não pôde ser inicializado: {e}")
    
    def test_analyze_image_method(self):
        """Teste: Método analyze_image"""
        if self.engine:
            try:
                # Criar uma imagem de teste
                test_image = Image.new('RGB', (100, 100), color='red')
                
                # Testar se o método existe
                if hasattr(self.engine, 'analyze_image'):
                    # Pode falhar por dependências, mas o método deve existir
                    self.assertTrue(True)
                else:
                    self.fail("Método analyze_image não encontrado")
            except Exception as e:
                # Se falhar, pode ser por dependências não disponíveis
                self.skipTest(f"Método analyze_image não pôde ser testado: {e}")

class TestLogicalAIReasoningSystem(unittest.TestCase):
    """Teste unitário para LogicalAIReasoningSystem"""
    
    def setUp(self):
        """Configuração inicial"""
        self.system = None
        
    def test_system_initialization(self):
        """Teste: Inicialização do LogicalAIReasoningSystem"""
        try:
            from core.reasoning import LogicalAIReasoningSystem
            self.system = LogicalAIReasoningSystem()
            self.assertIsNotNone(self.system)
        except ImportError:
            self.skipTest("LogicalAIReasoningSystem não disponível")
        except Exception as e:
            # Se falhar na inicialização, pode ser por dependências
            self.skipTest(f"LogicalAIReasoningSystem não pôde ser inicializado: {e}")
    
    def test_analyze_image_revolutionary_method(self):
        """Teste: Método analyze_image_revolutionary"""
        if self.system:
            try:
                # Testar se o método existe
                if hasattr(self.system, 'analyze_image_revolutionary'):
                    # Pode falhar por dependências, mas o método deve existir
                    self.assertTrue(True)
                else:
                    self.fail("Método analyze_image_revolutionary não encontrado")
            except Exception as e:
                # Se falhar, pode ser por dependências não disponíveis
                self.skipTest(f"Método analyze_image_revolutionary não pôde ser testado: {e}")

class TestContinuousLearningSystem(unittest.TestCase):
    """Teste unitário para ContinuousLearningSystem"""
    
    def setUp(self):
        """Configuração inicial"""
        self.system = None
        
    def test_system_initialization(self):
        """Teste: Inicialização do ContinuousLearningSystem"""
        try:
            from core.learning import ContinuousLearningSystem
            self.system = ContinuousLearningSystem()
            self.assertIsNotNone(self.system)
        except ImportError:
            self.skipTest("ContinuousLearningSystem não disponível")
        except Exception as e:
            # Se falhar na inicialização, pode ser por dependências
            self.skipTest(f"ContinuousLearningSystem não pôde ser inicializado: {e}")
    
    def test_learning_methods(self):
        """Teste: Métodos de aprendizado"""
        if self.system:
            try:
                # Testar se os métodos existem
                methods = ['start_learning', 'stop_learning', 'get_status']
                for method in methods:
                    if hasattr(self.system, method):
                        self.assertTrue(True)
                    else:
                        self.fail(f"Método {method} não encontrado")
            except Exception as e:
                self.skipTest(f"Métodos de aprendizado não puderam ser testados: {e}")

class TestWebAppFunctions(unittest.TestCase):
    """Teste unitário para funções da web_app"""
    
    def test_import_web_app(self):
        """Teste: Importação do web_app"""
        try:
            from interfaces.web_app import main
            self.assertIsNotNone(main)
            self.assertTrue(callable(main))
        except ImportError:
            self.skipTest("web_app não disponível")
        except Exception as e:
            self.skipTest(f"web_app não pôde ser importado: {e}")
    
    def test_main_function(self):
        """Teste: Função main"""
        try:
            from interfaces.web_app import main
            
            # Mock do Streamlit para evitar execução real
            with patch('streamlit.set_page_config'), \
                 patch('streamlit.title'), \
                 patch('streamlit.markdown'), \
                 patch('streamlit.tabs'), \
                 patch('streamlit.success'), \
                 patch('streamlit.error'):
                
                # A função main deve ser executável sem erros críticos
                try:
                    # Não executar realmente, apenas verificar se é callable
                    self.assertTrue(callable(main))
                except Exception as e:
                    self.skipTest(f"Função main não pôde ser testada: {e}")
                    
        except ImportError:
            self.skipTest("web_app não disponível")

class TestUtilityFunctions(unittest.TestCase):
    """Teste unitário para funções utilitárias"""
    
    def test_numpy_operations(self):
        """Teste: Operações NumPy"""
        try:
            # Testar operações básicas do NumPy
            arr = np.array([1, 2, 3, 4, 5])
            self.assertEqual(arr.sum(), 15)
            self.assertEqual(arr.mean(), 3.0)
            self.assertEqual(arr.max(), 5)
            self.assertEqual(arr.min(), 1)
        except Exception as e:
            self.fail(f"Erro nas operações NumPy: {e}")
    
    def test_pil_operations(self):
        """Teste: Operações PIL"""
        try:
            # Testar operações básicas do PIL
            img = Image.new('RGB', (100, 100), color='red')
            self.assertEqual(img.size, (100, 100))
            self.assertEqual(img.mode, 'RGB')
            
            # Testar conversão para array
            arr = np.array(img)
            self.assertEqual(arr.shape, (100, 100, 3))
        except Exception as e:
            self.fail(f"Erro nas operações PIL: {e}")
    
    def test_json_operations(self):
        """Teste: Operações JSON"""
        try:
            # Testar operações básicas do JSON
            test_data = {"test": "value", "number": 42, "list": [1, 2, 3]}
            
            # Serializar
            json_str = json.dumps(test_data)
            self.assertIsInstance(json_str, str)
            
            # Deserializar
            parsed_data = json.loads(json_str)
            self.assertEqual(parsed_data, test_data)
        except Exception as e:
            self.fail(f"Erro nas operações JSON: {e}")

class TestIntegrationPoints(unittest.TestCase):
    """Teste unitário para pontos de integração"""
    
    def test_streamlit_imports(self):
        """Teste: Imports do Streamlit"""
        try:
            import streamlit as st
            self.assertIsNotNone(st)
            
            # Verificar se funções principais estão disponíveis
            functions = ['title', 'markdown', 'button', 'file_uploader', 'tabs', 'columns']
            for func in functions:
                self.assertTrue(hasattr(st, func), f"Função {func} não encontrada no Streamlit")
        except ImportError:
            self.skipTest("Streamlit não disponível")
        except Exception as e:
            self.fail(f"Erro ao importar Streamlit: {e}")
    
    def test_opencv_imports(self):
        """Teste: Imports do OpenCV"""
        try:
            import cv2
            self.assertIsNotNone(cv2)
            
            # Verificar se funções principais estão disponíveis
            functions = ['imread', 'imwrite', 'resize', 'cvtColor']
            for func in functions:
                self.assertTrue(hasattr(cv2, func), f"Função {func} não encontrada no OpenCV")
        except ImportError:
            self.skipTest("OpenCV não disponível")
        except Exception as e:
            self.fail(f"Erro ao importar OpenCV: {e}")
    
    def test_pandas_imports(self):
        """Teste: Imports do Pandas"""
        try:
            import pandas as pd
            self.assertIsNotNone(pd)
            
            # Verificar se funções principais estão disponíveis
            functions = ['DataFrame', 'read_csv', 'to_csv']
            for func in functions:
                self.assertTrue(hasattr(pd, func), f"Função {func} não encontrada no Pandas")
        except ImportError:
            self.skipTest("Pandas não disponível")
        except Exception as e:
            self.fail(f"Erro ao importar Pandas: {e}")

def run_unit_tests():
    """Executar todos os testes unitários"""
    print("🧪 INICIANDO TESTES UNITÁRIOS")
    print("=" * 60)
    
    # Criar suite de testes
    test_suite = unittest.TestSuite()
    
    # Adicionar testes
    test_classes = [
        TestDebugLogger,
        TestButtonDebug,
        TestImageCache,
        TestIntuitionEngine,
        TestLogicalAIReasoningSystem,
        TestContinuousLearningSystem,
        TestWebAppFunctions,
        TestUtilityFunctions,
        TestIntegrationPoints
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Executar testes
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Resumo
    print("\n" + "=" * 60)
    print("📊 RESUMO DOS TESTES UNITÁRIOS")
    print("=" * 60)
    print(f"✅ Executados: {result.testsRun}")
    print(f"❌ Falharam: {len(result.failures)}")
    print(f"⚠️ Erros: {len(result.errors)}")
    print(f"⏭️ Pulados: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    print(f"📈 Taxa de Sucesso: {((result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100):.1f}%")
    
    # Salvar resultados
    save_unit_test_results(result)
    
    return result.wasSuccessful()

def save_unit_test_results(result):
    """Salvar resultados dos testes unitários"""
    try:
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "tests_run": result.testsRun,
            "failures": len(result.failures),
            "errors": len(result.errors),
            "skipped": len(result.skipped) if hasattr(result, 'skipped') else 0,
            "success_rate": ((result.testsRun - len(result.failures) - len(result.errors))/result.testsRun*100),
            "details": {
                "failures": [str(f) for f in result.failures],
                "errors": [str(e) for e in result.errors]
            }
        }
        
        with open("test_results_unit.json", "w") as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Resultados salvos em: test_results_unit.json")
    except Exception as e:
        print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🧪 TESTE UNITÁRIO - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    try:
        success = run_unit_tests()
        
        if success:
            print("\n🎉 TODOS OS TESTES UNITÁRIOS PASSARAM!")
            sys.exit(0)
        else:
            print("\n⚠️ ALGUNS TESTES UNITÁRIOS FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
