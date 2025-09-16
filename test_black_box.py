#!/usr/bin/env python3
"""
Teste de Caixa Preta - Sistema de Identificação de Pássaros
Testa funcionalidades sem conhecimento interno do sistema
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
from PIL import Image
import io
import base64

class BlackBoxTester:
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
    
    def test_interface_elements(self):
        """Teste 1: Elementos da interface"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos essenciais da interface
            interface_elements = [
                "streamlit",
                "tabs",
                "sidebar",
                "button",
                "upload"
            ]
            
            found_elements = [elem for elem in interface_elements if elem in content.lower()]
            
            if len(found_elements) >= 3:
                self.log_test("Elementos da Interface", "PASS", f"Encontrados: {found_elements}")
                return True
            else:
                self.log_test("Elementos da Interface", "FAIL", f"Encontrados apenas: {found_elements}")
                return False
        except Exception as e:
            self.log_test("Elementos da Interface", "FAIL", str(e))
            return False
    
    def test_navigation_tabs(self):
        """Teste 2: Navegação entre tabs"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar se há múltiplas tabs
            tab_count = content.count('data-baseweb="tab"')
            
            if tab_count >= 5:
                self.log_test("Navegação entre Tabs", "PASS", f"{tab_count} tabs encontradas")
                return True
            else:
                self.log_test("Navegação entre Tabs", "FAIL", f"Apenas {tab_count} tabs encontradas")
                return False
        except Exception as e:
            self.log_test("Navegação entre Tabs", "FAIL", str(e))
            return False
    
    def test_file_upload_interface(self):
        """Teste 3: Interface de upload de arquivo"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de upload
            upload_elements = [
                "file_uploader",
                "upload",
                "input",
                "type="
            ]
            
            found_elements = [elem for elem in upload_elements if elem in content.lower()]
            
            if len(found_elements) >= 2:
                self.log_test("Interface de Upload", "PASS", f"Elementos encontrados: {found_elements}")
                return True
            else:
                self.log_test("Interface de Upload", "FAIL", f"Elementos insuficientes: {found_elements}")
                return False
        except Exception as e:
            self.log_test("Interface de Upload", "FAIL", str(e))
            return False
    
    def test_responsive_layout(self):
        """Teste 4: Layout responsivo"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar CSS responsivo
            responsive_elements = [
                "media query",
                "max-width",
                "flex",
                "responsive",
                "mobile"
            ]
            
            found_elements = [elem for elem in responsive_elements if elem in content.lower()]
            
            if len(found_elements) >= 2:
                self.log_test("Layout Responsivo", "PASS", f"Elementos CSS encontrados: {found_elements}")
                return True
            else:
                self.log_test("Layout Responsivo", "FAIL", f"Elementos CSS insuficientes: {found_elements}")
                return False
        except Exception as e:
            self.log_test("Layout Responsivo", "FAIL", str(e))
            return False
    
    def test_error_handling_ui(self):
        """Teste 5: Tratamento de erros na UI"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de tratamento de erro
            error_elements = [
                "error",
                "exception",
                "try",
                "catch",
                "alert"
            ]
            
            found_elements = [elem for elem in error_elements if elem in content.lower()]
            
            if len(found_elements) >= 1:
                self.log_test("Tratamento de Erros UI", "PASS", f"Elementos encontrados: {found_elements}")
                return True
            else:
                self.log_test("Tratamento de Erros UI", "WARN", "Nenhum elemento de erro encontrado")
                return False
        except Exception as e:
            self.log_test("Tratamento de Erros UI", "FAIL", str(e))
            return False
    
    def test_data_visualization(self):
        """Teste 6: Visualização de dados"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de visualização
            viz_elements = [
                "plotly",
                "chart",
                "graph",
                "metric",
                "dataframe"
            ]
            
            found_elements = [elem for elem in viz_elements if elem in content.lower()]
            
            if len(found_elements) >= 2:
                self.log_test("Visualização de Dados", "PASS", f"Elementos encontrados: {found_elements}")
                return True
            else:
                self.log_test("Visualização de Dados", "FAIL", f"Elementos insuficientes: {found_elements}")
                return False
        except Exception as e:
            self.log_test("Visualização de Dados", "FAIL", str(e))
            return False
    
    def test_user_interaction_elements(self):
        """Teste 7: Elementos de interação do usuário"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de interação
            interaction_elements = [
                "button",
                "click",
                "select",
                "input",
                "form"
            ]
            
            found_elements = [elem for elem in interaction_elements if elem in content.lower()]
            
            if len(found_elements) >= 3:
                self.log_test("Elementos de Interação", "PASS", f"Elementos encontrados: {found_elements}")
                return True
            else:
                self.log_test("Elementos de Interação", "FAIL", f"Elementos insuficientes: {found_elements}")
                return False
        except Exception as e:
            self.log_test("Elementos de Interação", "FAIL", str(e))
            return False
    
    def test_accessibility_features(self):
        """Teste 8: Recursos de acessibilidade"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de acessibilidade
            accessibility_elements = [
                "alt=",
                "title=",
                "aria-",
                "role=",
                "label"
            ]
            
            found_elements = [elem for elem in accessibility_elements if elem in content.lower()]
            
            if len(found_elements) >= 1:
                self.log_test("Recursos de Acessibilidade", "PASS", f"Elementos encontrados: {found_elements}")
                return True
            else:
                self.log_test("Recursos de Acessibilidade", "WARN", "Nenhum elemento de acessibilidade encontrado")
                return False
        except Exception as e:
            self.log_test("Recursos de Acessibilidade", "FAIL", str(e))
            return False
    
    def test_performance_indicators(self):
        """Teste 9: Indicadores de performance"""
        try:
            start_time = time.time()
            response = requests.get(self.base_url, timeout=30)
            end_time = time.time()
            
            response_time = end_time - start_time
            content_size = len(response.content)
            
            # Verificar se há indicadores de performance na página
            performance_elements = [
                "loading",
                "spinner",
                "progress",
                "timeout"
            ]
            
            found_elements = [elem for elem in performance_elements if elem in response.text.lower()]
            
            if response_time < 5 and content_size > 1000:
                self.log_test("Indicadores de Performance", "PASS", f"Tempo: {response_time:.2f}s, Tamanho: {content_size} bytes")
                return True
            else:
                self.log_test("Indicadores de Performance", "FAIL", f"Tempo: {response_time:.2f}s, Tamanho: {content_size} bytes")
                return False
        except Exception as e:
            self.log_test("Indicadores de Performance", "FAIL", str(e))
            return False
    
    def test_security_indicators(self):
        """Teste 10: Indicadores de segurança"""
        try:
            response = requests.get(self.base_url, timeout=10)
            
            # Verificar cabeçalhos de segurança
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Content-Security-Policy"
            ]
            
            found_headers = [h for h in security_headers if h in response.headers]
            
            # Verificar se não há informações sensíveis expostas
            content = response.text.lower()
            sensitive_info = [
                "password",
                "secret",
                "key",
                "token",
                "api_key"
            ]
            
            exposed_info = [info for info in sensitive_info if info in content]
            
            if len(found_headers) >= 0 and len(exposed_info) == 0:
                self.log_test("Indicadores de Segurança", "PASS", f"Cabeçalhos: {len(found_headers)}, Info exposta: {len(exposed_info)}")
                return True
            else:
                self.log_test("Indicadores de Segurança", "WARN", f"Cabeçalhos: {len(found_headers)}, Info exposta: {exposed_info}")
                return False
        except Exception as e:
            self.log_test("Indicadores de Segurança", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de caixa preta"""
        print("🖤 INICIANDO TESTES DE CAIXA PRETA")
        print("=" * 60)
        
        tests = [
            self.test_interface_elements,
            self.test_navigation_tabs,
            self.test_file_upload_interface,
            self.test_responsive_layout,
            self.test_error_handling_ui,
            self.test_data_visualization,
            self.test_user_interaction_elements,
            self.test_accessibility_features,
            self.test_performance_indicators,
            self.test_security_indicators
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
        print("📊 RESUMO DOS TESTES DE CAIXA PRETA")
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
            with open("test_results_black_box.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_black_box.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🖤 TESTE DE CAIXA PRETA - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = BlackBoxTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE CAIXA PRETA PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE CAIXA PRETA FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
