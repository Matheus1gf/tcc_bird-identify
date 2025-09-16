#!/usr/bin/env python3
"""
Teste de Sistema - Sistema de Identificação de Pássaros
Testa o sistema completo como um todo
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
import subprocess
import threading
import signal

class SystemTester:
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
    
    def test_system_startup(self):
        """Teste 1: Inicialização do sistema"""
        try:
            # Verificar se o sistema pode ser iniciado
            startup_success = 0
            total_checks = 0
            
            # Verificar se main.py existe
            total_checks += 1
            if os.path.exists("main.py"):
                startup_success += 1
            
            # Verificar se src/interfaces/web_app.py existe
            total_checks += 1
            if os.path.exists("src/interfaces/web_app.py"):
                startup_success += 1
            
            # Verificar se requirements.txt existe
            total_checks += 1
            if os.path.exists("requirements.txt"):
                startup_success += 1
            
            if startup_success >= 2:
                self.log_test("Inicialização do Sistema", "PASS", f"{startup_success}/{total_checks} componentes encontrados")
                return True
            else:
                self.log_test("Inicialização do Sistema", "FAIL", f"Apenas {startup_success}/{total_checks} componentes encontrados")
                return False
        except Exception as e:
            self.log_test("Inicialização do Sistema", "FAIL", str(e))
            return False
    
    def test_system_availability(self):
        """Teste 2: Disponibilidade do sistema"""
        try:
            # Testar se o sistema está disponível
            response = requests.get(self.base_url, timeout=10)
            
            if response.status_code == 200:
                self.log_test("Disponibilidade do Sistema", "PASS", f"HTTP {response.status_code}")
                return True
            else:
                self.log_test("Disponibilidade do Sistema", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Disponibilidade do Sistema", "FAIL", str(e))
            return False
    
    def test_system_performance(self):
        """Teste 3: Performance do sistema"""
        try:
            # Testar performance do sistema
            start_time = time.time()
            response = requests.get(self.base_url, timeout=30)
            end_time = time.time()
            
            response_time = end_time - start_time
            content_size = len(response.content)
            
            if response_time < 10 and content_size > 1000:
                self.log_test("Performance do Sistema", "PASS", f"Tempo: {response_time:.2f}s, Tamanho: {content_size} bytes")
                return True
            else:
                self.log_test("Performance do Sistema", "FAIL", f"Tempo: {response_time:.2f}s, Tamanho: {content_size} bytes")
                return False
        except Exception as e:
            self.log_test("Performance do Sistema", "FAIL", str(e))
            return False
    
    def test_system_stability(self):
        """Teste 4: Estabilidade do sistema"""
        try:
            # Testar estabilidade com múltiplas requisições
            success_count = 0
            total_requests = 5
            
            for i in range(total_requests):
                try:
                    response = requests.get(self.base_url, timeout=5)
                    if response.status_code == 200:
                        success_count += 1
                    time.sleep(1)
                except:
                    pass
            
            if success_count >= 4:
                self.log_test("Estabilidade do Sistema", "PASS", f"{success_count}/{total_requests} requisições bem-sucedidas")
                return True
            else:
                self.log_test("Estabilidade do Sistema", "FAIL", f"Apenas {success_count}/{total_requests} requisições bem-sucedidas")
                return False
        except Exception as e:
            self.log_test("Estabilidade do Sistema", "FAIL", str(e))
            return False
    
    def test_system_functionality(self):
        """Teste 5: Funcionalidade do sistema"""
        try:
            # Testar funcionalidades básicas do sistema
            functionality_indicators = 0
            
            # Verificar se há funcionalidades básicas
            if os.path.exists("src/interfaces/web_app.py"):
                with open("src/interfaces/web_app.py", 'r') as f:
                    content = f.read()
                
                # Verificar funcionalidades
                features = [
                    "st.title",
                    "st.tabs",
                    "st.button",
                    "st.file_uploader",
                    "st.image",
                    "st.metric"
                ]
                
                for feature in features:
                    if feature in content:
                        functionality_indicators += 1
            
            if functionality_indicators >= 4:
                self.log_test("Funcionalidade do Sistema", "PASS", f"{functionality_indicators} funcionalidades encontradas")
                return True
            else:
                self.log_test("Funcionalidade do Sistema", "FAIL", f"Apenas {functionality_indicators} funcionalidades encontradas")
                return False
        except Exception as e:
            self.log_test("Funcionalidade do Sistema", "FAIL", str(e))
            return False
    
    def test_system_security(self):
        """Teste 6: Segurança do sistema"""
        try:
            # Testar aspectos de segurança
            security_indicators = 0
            
            # Verificar se há implementações de segurança
            if os.path.exists("src/interfaces/web_app.py"):
                with open("src/interfaces/web_app.py", 'r') as f:
                    content = f.read()
                
                # Verificar padrões de segurança
                security_patterns = [
                    "try:",
                    "except",
                    "st.error",
                    "debug_logger.log_error"
                ]
                
                for pattern in security_patterns:
                    if pattern in content:
                        security_indicators += 1
            
            # Verificar se não há padrões inseguros
            insecure_patterns = [
                "eval(",
                "exec(",
                "os.system("
            ]
            
            insecure_found = 0
            if os.path.exists("src/interfaces/web_app.py"):
                with open("src/interfaces/web_app.py", 'r') as f:
                    content = f.read()
                
                for pattern in insecure_patterns:
                    if pattern in content:
                        insecure_found += 1
            
            if security_indicators >= 2 and insecure_found == 0:
                self.log_test("Segurança do Sistema", "PASS", f"{security_indicators} indicadores de segurança, {insecure_found} padrões inseguros")
                return True
            else:
                self.log_test("Segurança do Sistema", "WARN", f"{security_indicators} indicadores de segurança, {insecure_found} padrões inseguros")
                return False
        except Exception as e:
            self.log_test("Segurança do Sistema", "FAIL", str(e))
            return False
    
    def test_system_scalability(self):
        """Teste 7: Escalabilidade do sistema"""
        try:
            # Testar escalabilidade com múltiplas requisições simultâneas
            success_count = 0
            total_requests = 10
            
            def make_request():
                nonlocal success_count
                try:
                    response = requests.get(self.base_url, timeout=5)
                    if response.status_code == 200:
                        success_count += 1
                except:
                    pass
            
            # Fazer requisições simultâneas
            threads = []
            for i in range(total_requests):
                thread = threading.Thread(target=make_request)
                threads.append(thread)
                thread.start()
            
            # Aguardar todas as threads
            for thread in threads:
                thread.join()
            
            if success_count >= 8:
                self.log_test("Escalabilidade do Sistema", "PASS", f"{success_count}/{total_requests} requisições simultâneas bem-sucedidas")
                return True
            else:
                self.log_test("Escalabilidade do Sistema", "FAIL", f"Apenas {success_count}/{total_requests} requisições simultâneas bem-sucedidas")
                return False
        except Exception as e:
            self.log_test("Escalabilidade do Sistema", "FAIL", str(e))
            return False
    
    def test_system_reliability(self):
        """Teste 8: Confiabilidade do sistema"""
        try:
            # Testar confiabilidade com requisições repetidas
            success_count = 0
            total_requests = 20
            
            for i in range(total_requests):
                try:
                    response = requests.get(self.base_url, timeout=5)
                    if response.status_code == 200:
                        success_count += 1
                    time.sleep(0.5)
                except:
                    pass
            
            reliability_rate = success_count / total_requests
            
            if reliability_rate >= 0.9:
                self.log_test("Confiabilidade do Sistema", "PASS", f"Taxa de confiabilidade: {reliability_rate:.2%}")
                return True
            else:
                self.log_test("Confiabilidade do Sistema", "FAIL", f"Taxa de confiabilidade: {reliability_rate:.2%}")
                return False
        except Exception as e:
            self.log_test("Confiabilidade do Sistema", "FAIL", str(e))
            return False
    
    def test_system_maintainability(self):
        """Teste 9: Manutenibilidade do sistema"""
        try:
            # Testar manutenibilidade do código
            maintainability_indicators = 0
            
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
                    
                    # Verificar padrões de manutenibilidade
                    patterns = [
                        "def ",
                        "class ",
                        "# ",
                        '"""',
                        "docstring"
                    ]
                    
                    for pattern in patterns:
                        if pattern in content:
                            maintainability_indicators += 1
            
            if maintainability_indicators >= 10:
                self.log_test("Manutenibilidade do Sistema", "PASS", f"{maintainability_indicators} indicadores encontrados")
                return True
            else:
                self.log_test("Manutenibilidade do Sistema", "FAIL", f"Apenas {maintainability_indicators} indicadores encontrados")
                return False
        except Exception as e:
            self.log_test("Manutenibilidade do Sistema", "FAIL", str(e))
            return False
    
    def test_system_usability(self):
        """Teste 10: Usabilidade do sistema"""
        try:
            # Testar usabilidade do sistema
            usability_indicators = 0
            
            # Verificar se há elementos de usabilidade
            if os.path.exists("src/interfaces/web_app.py"):
                with open("src/interfaces/web_app.py", 'r') as f:
                    content = f.read()
                
                # Verificar elementos de usabilidade
                usability_elements = [
                    "st.title",
                    "st.markdown",
                    "st.info",
                    "st.success",
                    "st.warning",
                    "st.error",
                    "st.help",
                    "st.tooltip"
                ]
                
                for element in usability_elements:
                    if element in content:
                        usability_indicators += 1
            
            if usability_indicators >= 4:
                self.log_test("Usabilidade do Sistema", "PASS", f"{usability_indicators} elementos de usabilidade encontrados")
                return True
            else:
                self.log_test("Usabilidade do Sistema", "FAIL", f"Apenas {usability_indicators} elementos de usabilidade encontrados")
                return False
        except Exception as e:
            self.log_test("Usabilidade do Sistema", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de sistema"""
        print("🖥️ INICIANDO TESTES DE SISTEMA")
        print("=" * 60)
        
        tests = [
            self.test_system_startup,
            self.test_system_availability,
            self.test_system_performance,
            self.test_system_stability,
            self.test_system_functionality,
            self.test_system_security,
            self.test_system_scalability,
            self.test_system_reliability,
            self.test_system_maintainability,
            self.test_system_usability
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
            
            time.sleep(1)  # Pausa entre testes
        
        # Resumo
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES DE SISTEMA")
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
            with open("test_results_system.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_system.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🖥️ TESTE DE SISTEMA - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = SystemTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE SISTEMA PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE SISTEMA FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
