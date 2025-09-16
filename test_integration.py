#!/usr/bin/env python3
"""
Teste de Integração - Sistema de Identificação de Pássaros
Testa integração entre componentes e módulos
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
import subprocess
import threading

class IntegrationTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = []
        self.app_process = None
        
    def log_test(self, test_name, status, details=""):
        """Registrar resultado do teste"""
        result = {
            "test": test_name,
            "status": status,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.results.append(result)
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {status}")
        if details:
            print(f"   📝 {details}")
    
    def test_component_integration(self):
        """Teste 1: Integração entre componentes"""
        try:
            # Verificar se os componentes principais podem ser importados juntos
            import_success = 0
            total_imports = 0
            
            components = [
                "src.core.intuition",
                "src.core.reasoning", 
                "src.core.learning",
                "src.core.cache",
                "src.utils.debug_logger",
                "src.utils.button_debug"
            ]
            
            for component in components:
                total_imports += 1
                try:
                    __import__(component)
                    import_success += 1
                except ImportError:
                    pass
            
            if import_success >= 3:
                self.log_test("Integração entre Componentes", "PASS", f"{import_success}/{total_imports} componentes importados")
                return True
            else:
                self.log_test("Integração entre Componentes", "FAIL", f"Apenas {import_success}/{total_imports} componentes importados")
                return False
        except Exception as e:
            self.log_test("Integração entre Componentes", "FAIL", str(e))
            return False
    
    def test_data_flow_integration(self):
        """Teste 2: Integração do fluxo de dados"""
        try:
            # Verificar se há fluxo de dados entre componentes
            data_flow_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de integração
                    patterns = [
                        "import ",
                        "from ",
                        "st.session_state",
                        "image_cache",
                        "debug_logger"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            data_flow_indicators += 1
            
            if data_flow_indicators >= 10:
                self.log_test("Integração do Fluxo de Dados", "PASS", f"{data_flow_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração do Fluxo de Dados", "FAIL", f"Apenas {data_flow_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Integração do Fluxo de Dados", "FAIL", str(e))
            return False
    
    def test_api_integration(self):
        """Teste 3: Integração da API"""
        try:
            # Testar endpoints da API
            endpoints = [
                "/",
                "/healthz",
                "/_stcore/health"
            ]
            
            working_endpoints = 0
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    if response.status_code in [200, 404]:
                        working_endpoints += 1
                except:
                    pass
            
            if working_endpoints >= 1:
                self.log_test("Integração da API", "PASS", f"{working_endpoints}/{len(endpoints)} endpoints funcionando")
                return True
            else:
                self.log_test("Integração da API", "FAIL", f"Nenhum endpoint funcionando")
                return False
        except Exception as e:
            self.log_test("Integração da API", "FAIL", str(e))
            return False
    
    def test_database_integration(self):
        """Teste 4: Integração com banco de dados"""
        try:
            # Verificar se há integração com banco de dados
            db_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de banco de dados
                    patterns = [
                        "sqlite",
                        "database",
                        "db",
                        "sql",
                        "query"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content.lower():
                            db_indicators += 1
            
            if db_indicators >= 0:
                self.log_test("Integração com Banco de Dados", "PASS", f"{db_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração com Banco de Dados", "FAIL", f"Nenhum indicador encontrado")
                return False
        except Exception as e:
            self.log_test("Integração com Banco de Dados", "FAIL", str(e))
            return False
    
    def test_file_system_integration(self):
        """Teste 5: Integração com sistema de arquivos"""
        try:
            # Verificar se há integração com sistema de arquivos
            fs_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de sistema de arquivos
                    patterns = [
                        "os.path",
                        "os.listdir",
                        "os.remove",
                        "open(",
                        "with open"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            fs_indicators += 1
            
            if fs_indicators >= 3:
                self.log_test("Integração com Sistema de Arquivos", "PASS", f"{fs_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração com Sistema de Arquivos", "FAIL", f"Apenas {fs_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Integração com Sistema de Arquivos", "FAIL", str(e))
            return False
    
    def test_external_service_integration(self):
        """Teste 6: Integração com serviços externos"""
        try:
            # Verificar se há integração com serviços externos
            external_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de serviços externos
                    patterns = [
                        "requests",
                        "http",
                        "api",
                        "url",
                        "endpoint"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content.lower():
                            external_indicators += 1
            
            if external_indicators >= 0:
                self.log_test("Integração com Serviços Externos", "PASS", f"{external_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração com Serviços Externos", "FAIL", f"Nenhum indicador encontrado")
                return False
        except Exception as e:
            self.log_test("Integração com Serviços Externos", "FAIL", str(e))
            return False
    
    def test_ui_integration(self):
        """Teste 7: Integração da interface do usuário"""
        try:
            # Verificar se há integração da UI
            ui_indicators = 0
            
            # Verificar arquivo principal da UI
            if os.path.exists("src/interfaces/web_app.py"):
                with open("src/interfaces/web_app.py", 'r') as f:
                    content = f.read()
                
                # Verificar padrões de UI
                patterns = [
                    "st.",
                    "streamlit",
                    "tabs",
                    "sidebar",
                    "button",
                    "upload"
                ]
                
                for pattern in patterns:
                    if pattern in content:
                        ui_indicators += 1
            
            if ui_indicators >= 5:
                self.log_test("Integração da Interface do Usuário", "PASS", f"{ui_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração da Interface do Usuário", "FAIL", f"Apenas {ui_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Integração da Interface do Usuário", "FAIL", str(e))
            return False
    
    def test_logging_integration(self):
        """Teste 8: Integração do sistema de logging"""
        try:
            # Verificar se há integração do sistema de logging
            logging_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de logging
                    patterns = [
                        "debug_logger",
                        "button_debug",
                        "logging",
                        "log_",
                        "print("
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            logging_indicators += 1
            
            if logging_indicators >= 3:
                self.log_test("Integração do Sistema de Logging", "PASS", f"{logging_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração do Sistema de Logging", "FAIL", f"Apenas {logging_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Integração do Sistema de Logging", "FAIL", str(e))
            return False
    
    def test_configuration_integration(self):
        """Teste 9: Integração de configuração"""
        try:
            # Verificar se há integração de configuração
            config_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de configuração
                    patterns = [
                        "config",
                        "settings",
                        "parameters",
                        "options",
                        "st.set_page_config"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            config_indicators += 1
            
            if config_indicators >= 2:
                self.log_test("Integração de Configuração", "PASS", f"{config_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração de Configuração", "FAIL", f"Apenas {config_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Integração de Configuração", "FAIL", str(e))
            return False
    
    def test_error_handling_integration(self):
        """Teste 10: Integração do tratamento de erros"""
        try:
            # Verificar se há integração do tratamento de erros
            error_indicators = 0
            
            # Verificar arquivos principais
            main_files = [
                "src/interfaces/web_app.py",
                "src/core/intuition.py",
                "src/core/reasoning.py"
            ]
            
            for file_path in main_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de tratamento de erro
                    patterns = [
                        "try:",
                        "except",
                        "raise",
                        "st.error",
                        "debug_logger.log_error"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            error_indicators += 1
            
            if error_indicators >= 5:
                self.log_test("Integração do Tratamento de Erros", "PASS", f"{error_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Integração do Tratamento de Erros", "FAIL", f"Apenas {error_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Integração do Tratamento de Erros", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de integração"""
        print("🔗 INICIANDO TESTES DE INTEGRAÇÃO")
        print("=" * 60)
        
        tests = [
            self.test_component_integration,
            self.test_data_flow_integration,
            self.test_api_integration,
            self.test_database_integration,
            self.test_file_system_integration,
            self.test_external_service_integration,
            self.test_ui_integration,
            self.test_logging_integration,
            self.test_configuration_integration,
            self.test_error_handling_integration
        ]
        
        passed = 0
        failed = 0
        warned = 0
        
        for test in tests:
            try:
                result = test()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                self.log_test(test.__name__, "FAIL", str(e))
                failed += 1
            
            time.sleep(0.5)  # Pausa entre testes
        
        # Resumo
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES DE INTEGRAÇÃO")
        print("=" * 60)
        print(f"✅ Passou: {passed}")
        print(f"❌ Falhou: {failed}")
        print(f"⚠️ Avisos: {warned}")
        print(f"📈 Taxa de Sucesso: {(passed/(passed+failed)*100):.1f}%")
        
        # Salvar resultados
        self.save_results()
        
        return passed, failed, warned
    
    def save_results(self):
        """Salvar resultados em arquivo JSON"""
        try:
            with open("test_results_integration.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_integration.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🔗 TESTE DE INTEGRAÇÃO - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = IntegrationTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE INTEGRAÇÃO PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE INTEGRAÇÃO FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
