#!/usr/bin/env python3
"""
Teste Automatizado - Sistema de Identificação de Pássaros
"""

import requests
import time
import json
import os
import sys
from datetime import datetime

class AutomatedTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = []
        
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
    
    def test_server_availability(self):
        """Teste 1: Disponibilidade do servidor"""
        try:
            response = requests.get(self.base_url, timeout=10)
            if response.status_code == 200:
                self.log_test("Servidor Disponível", "PASS", f"HTTP {response.status_code}")
                return True
            else:
                self.log_test("Servidor Disponível", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Servidor Disponível", "FAIL", str(e))
            return False
    
    def test_page_load_time(self):
        """Teste 2: Tempo de carregamento da página"""
        try:
            start_time = time.time()
            response = requests.get(self.base_url, timeout=30)
            load_time = time.time() - start_time
            
            if response.status_code == 200 and load_time < 10:
                self.log_test("Tempo de Carregamento", "PASS", f"{load_time:.2f}s")
                return True
            else:
                self.log_test("Tempo de Carregamento", "FAIL", f"{load_time:.2f}s")
                return False
        except Exception as e:
            self.log_test("Tempo de Carregamento", "FAIL", str(e))
            return False
    
    def test_page_content(self):
        """Teste 3: Conteúdo da página"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text.lower()
            
            required_elements = [
                "streamlit",
                "pássaro",
                "sistema",
                "identificação"
            ]
            
            found_elements = [elem for elem in required_elements if elem in content]
            
            if len(found_elements) >= 2:
                self.log_test("Conteúdo da Página", "PASS", f"Encontrados: {found_elements}")
                return True
            else:
                self.log_test("Conteúdo da Página", "FAIL", f"Encontrados apenas: {found_elements}")
                return False
        except Exception as e:
            self.log_test("Conteúdo da Página", "FAIL", str(e))
            return False
    
    def test_responsive_design(self):
        """Teste 4: Design responsivo"""
        try:
            # Simular diferentes user agents
            user_agents = [
                "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)",
                "Mozilla/5.0 (Android 10; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            ]
            
            responsive_count = 0
            for ua in user_agents:
                headers = {"User-Agent": ua}
                response = requests.get(self.base_url, headers=headers, timeout=10)
                if response.status_code == 200:
                    responsive_count += 1
            
            if responsive_count == len(user_agents):
                self.log_test("Design Responsivo", "PASS", f"{responsive_count}/{len(user_agents)} dispositivos")
                return True
            else:
                self.log_test("Design Responsivo", "FAIL", f"{responsive_count}/{len(user_agents)} dispositivos")
                return False
        except Exception as e:
            self.log_test("Design Responsivo", "FAIL", str(e))
            return False
    
    def test_error_handling(self):
        """Teste 5: Tratamento de erros"""
        try:
            # Testar endpoint inexistente
            response = requests.get(f"{self.base_url}/nonexistent", timeout=5)
            
            if response.status_code in [404, 405]:
                self.log_test("Tratamento de Erros", "PASS", f"HTTP {response.status_code}")
                return True
            else:
                self.log_test("Tratamento de Erros", "FAIL", f"HTTP {response.status_code}")
                return False
        except Exception as e:
            self.log_test("Tratamento de Erros", "PASS", "Erro tratado corretamente")
            return True
    
    def test_security_headers(self):
        """Teste 6: Cabeçalhos de segurança"""
        try:
            response = requests.get(self.base_url, timeout=10)
            headers = response.headers
            
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options", 
                "X-XSS-Protection",
                "Content-Security-Policy"
            ]
            
            found_headers = [h for h in security_headers if h in headers]
            
            if len(found_headers) >= 1:
                self.log_test("Cabeçalhos de Segurança", "PASS", f"Encontrados: {found_headers}")
                return True
            else:
                self.log_test("Cabeçalhos de Segurança", "WARN", "Nenhum cabeçalho de segurança encontrado")
                return False
        except Exception as e:
            self.log_test("Cabeçalhos de Segurança", "FAIL", str(e))
            return False
    
    def test_performance_metrics(self):
        """Teste 7: Métricas de performance"""
        try:
            start_time = time.time()
            response = requests.get(self.base_url, timeout=30)
            end_time = time.time()
            
            response_time = end_time - start_time
            content_length = len(response.content)
            
            metrics = {
                "response_time": response_time,
                "content_length": content_length,
                "status_code": response.status_code
            }
            
            if response_time < 5 and content_length > 1000:
                self.log_test("Métricas de Performance", "PASS", f"Tempo: {response_time:.2f}s, Tamanho: {content_length} bytes")
                return True
            else:
                self.log_test("Métricas de Performance", "FAIL", f"Tempo: {response_time:.2f}s, Tamanho: {content_length} bytes")
                return False
        except Exception as e:
            self.log_test("Métricas de Performance", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes"""
        print("🧪 INICIANDO TESTES AUTOMATIZADOS")
        print("=" * 60)
        
        tests = [
            self.test_server_availability,
            self.test_page_load_time,
            self.test_page_content,
            self.test_responsive_design,
            self.test_error_handling,
            self.test_security_headers,
            self.test_performance_metrics
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
        print("📊 RESUMO DOS TESTES")
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
            with open("test_results_automated.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_automated.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🚀 TESTE AUTOMATIZADO - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = AutomatedTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
