#!/usr/bin/env python3
"""
Teste de Caixa Cinza - Sistema de Identificação de Pássaros
Combina conhecimento interno e externo para testes mais abrangentes
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
import ast
import subprocess

class GrayBoxTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = []
        self.source_files = [
            "src/interfaces/web_app.py",
            "src/core/intuition.py",
            "src/core/reasoning.py",
            "src/core/learning.py"
        ]
        
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
    
    def test_api_endpoints(self):
        """Teste 1: Endpoints da API"""
        try:
            # Testar endpoints conhecidos do Streamlit
            endpoints = [
                "/",
                "/healthz",
                "/_stcore/health"
            ]
            
            working_endpoints = 0
            for endpoint in endpoints:
                try:
                    response = requests.get(f"{self.base_url}{endpoint}", timeout=5)
                    if response.status_code in [200, 404]:  # 404 é aceitável para alguns endpoints
                        working_endpoints += 1
                except:
                    pass
            
            if working_endpoints > 0:
                self.log_test("Endpoints da API", "PASS", f"{working_endpoints}/{len(endpoints)} endpoints funcionando")
                return True
            else:
                self.log_test("Endpoints da API", "FAIL", f"Nenhum endpoint funcionando")
                return False
        except Exception as e:
            self.log_test("Endpoints da API", "FAIL", str(e))
            return False
    
    def test_data_flow(self):
        """Teste 2: Fluxo de dados"""
        try:
            # Verificar se há fluxo de dados entre componentes
            data_flow_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de fluxo de dados
                    patterns = [
                        "st.session_state",
                        "st.cache",
                        "image_cache",
                        "debug_logger",
                        "button_debug"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            data_flow_indicators.append(f"{file_path}:{pattern}")
            
            if len(data_flow_indicators) >= 3:
                self.log_test("Fluxo de Dados", "PASS", f"{len(data_flow_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Fluxo de Dados", "FAIL", f"Apenas {len(data_flow_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Fluxo de Dados", "FAIL", str(e))
            return False
    
    def test_state_management(self):
        """Teste 3: Gerenciamento de estado"""
        try:
            state_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de gerenciamento de estado
                    patterns = [
                        "st.session_state",
                        "st.cache_data",
                        "st.experimental_memo",
                        "cache",
                        "state"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            state_indicators.append(f"{file_path}:{pattern}")
            
            if len(state_indicators) >= 2:
                self.log_test("Gerenciamento de Estado", "PASS", f"{len(state_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Gerenciamento de Estado", "FAIL", f"Apenas {len(state_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Gerenciamento de Estado", "FAIL", str(e))
            return False
    
    def test_error_propagation(self):
        """Teste 4: Propagação de erros"""
        try:
            error_propagation_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de propagação de erro
                    patterns = [
                        "except Exception",
                        "raise",
                        "st.error",
                        "debug_logger.log_error",
                        "try:"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            error_propagation_indicators.append(f"{file_path}:{pattern}")
            
            if len(error_propagation_indicators) >= 3:
                self.log_test("Propagação de Erros", "PASS", f"{len(error_propagation_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Propagação de Erros", "FAIL", f"Apenas {len(error_propagation_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Propagação de Erros", "FAIL", str(e))
            return False
    
    def test_performance_monitoring(self):
        """Teste 5: Monitoramento de performance"""
        try:
            performance_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de monitoramento de performance
                    patterns = [
                        "st.spinner",
                        "time.time()",
                        "datetime.now()",
                        "performance",
                        "timing"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            performance_indicators.append(f"{file_path}:{pattern}")
            
            if len(performance_indicators) >= 2:
                self.log_test("Monitoramento de Performance", "PASS", f"{len(performance_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Monitoramento de Performance", "FAIL", f"Apenas {len(performance_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Monitoramento de Performance", "FAIL", str(e))
            return False
    
    def test_security_implementation(self):
        """Teste 6: Implementação de segurança"""
        try:
            security_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de segurança
                    patterns = [
                        "sanitize",
                        "validate",
                        "escape",
                        "secure",
                        "auth"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            security_indicators.append(f"{file_path}:{pattern}")
            
            # Verificar se não há padrões inseguros
            insecure_patterns = [
                "eval(",
                "exec(",
                "os.system("
            ]
            
            insecure_found = []
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    for pattern in insecure_patterns:
                        if pattern in content:
                            insecure_found.append(f"{file_path}:{pattern}")
            
            if len(insecure_found) == 0:
                self.log_test("Implementação de Segurança", "PASS", f"Nenhum padrão inseguro encontrado")
                return True
            else:
                self.log_test("Implementação de Segurança", "WARN", f"Padrões inseguros: {insecure_found}")
                return False
        except Exception as e:
            self.log_test("Implementação de Segurança", "FAIL", str(e))
            return False
    
    def test_logging_system(self):
        """Teste 7: Sistema de logging"""
        try:
            logging_indicators = []
            
            for file_path in self.source_files:
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
                            logging_indicators.append(f"{file_path}:{pattern}")
            
            if len(logging_indicators) >= 3:
                self.log_test("Sistema de Logging", "PASS", f"{len(logging_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Sistema de Logging", "FAIL", f"Apenas {len(logging_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Sistema de Logging", "FAIL", str(e))
            return False
    
    def test_configuration_management(self):
        """Teste 8: Gerenciamento de configuração"""
        try:
            config_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de configuração
                    patterns = [
                        "st.set_page_config",
                        "config",
                        "settings",
                        "parameters",
                        "options"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            config_indicators.append(f"{file_path}:{pattern}")
            
            if len(config_indicators) >= 2:
                self.log_test("Gerenciamento de Configuração", "PASS", f"{len(config_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Gerenciamento de Configuração", "FAIL", f"Apenas {len(config_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Gerenciamento de Configuração", "FAIL", str(e))
            return False
    
    def test_resource_management(self):
        """Teste 9: Gerenciamento de recursos"""
        try:
            resource_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de gerenciamento de recursos
                    patterns = [
                        "with open(",
                        "close()",
                        "cleanup",
                        "temp_",
                        "os.remove"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            resource_indicators.append(f"{file_path}:{pattern}")
            
            if len(resource_indicators) >= 2:
                self.log_test("Gerenciamento de Recursos", "PASS", f"{len(resource_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Gerenciamento de Recursos", "FAIL", f"Apenas {len(resource_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Gerenciamento de Recursos", "FAIL", str(e))
            return False
    
    def test_integration_points(self):
        """Teste 10: Pontos de integração"""
        try:
            integration_indicators = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões de integração
                    patterns = [
                        "import ",
                        "from ",
                        "requests",
                        "api",
                        "endpoint"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            integration_indicators.append(f"{file_path}:{pattern}")
            
            if len(integration_indicators) >= 5:
                self.log_test("Pontos de Integração", "PASS", f"{len(integration_indicators)} indicadores encontrados")
                return True
            else:
                self.log_test("Pontos de Integração", "FAIL", f"Apenas {len(integration_indicators)} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Pontos de Integração", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de caixa cinza"""
        print("🔘 INICIANDO TESTES DE CAIXA CINZA")
        print("=" * 60)
        
        tests = [
            self.test_api_endpoints,
            self.test_data_flow,
            self.test_state_management,
            self.test_error_propagation,
            self.test_performance_monitoring,
            self.test_security_implementation,
            self.test_logging_system,
            self.test_configuration_management,
            self.test_resource_management,
            self.test_integration_points
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
        print("📊 RESUMO DOS TESTES DE CAIXA CINZA")
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
            with open("test_results_gray_box.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_gray_box.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🔘 TESTE DE CAIXA CINZA - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = GrayBoxTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE CAIXA CINZA PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE CAIXA CINZA FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
