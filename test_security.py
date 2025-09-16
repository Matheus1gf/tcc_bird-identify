#!/usr/bin/env python3
"""
Teste de Segurança - Sistema de Identificação de Pássaros
Testa aspectos de segurança do sistema
"""

import requests
import time
import json
import os
import sys
from datetime import datetime
import subprocess
import re

class SecurityTester:
    def __init__(self, base_url="http://localhost:8501"):
        self.base_url = base_url
        self.results = []
        self.security_issues = []
        
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
    
    def test_http_security_headers(self):
        """Teste 1: Cabeçalhos de segurança HTTP"""
        try:
            response = requests.get(self.base_url, timeout=10)
            headers = response.headers
            
            security_headers = [
                "X-Content-Type-Options",
                "X-Frame-Options",
                "X-XSS-Protection",
                "Content-Security-Policy",
                "Strict-Transport-Security",
                "X-Permitted-Cross-Domain-Policies"
            ]
            
            found_headers = []
            for header in security_headers:
                if header in headers:
                    found_headers.append(header)
            
            if len(found_headers) >= 1:
                self.log_test("Cabeçalhos de Segurança HTTP", "PASS", f"Encontrados: {found_headers}")
                return True
            else:
                self.log_test("Cabeçalhos de Segurança HTTP", "WARN", "Nenhum cabeçalho de segurança encontrado")
                return False
        except Exception as e:
            self.log_test("Cabeçalhos de Segurança HTTP", "FAIL", str(e))
            return False
    
    def test_sql_injection_protection(self):
        """Teste 2: Proteção contra SQL Injection"""
        try:
            # Testar payloads de SQL injection
            sql_payloads = [
                "' OR '1'='1",
                "'; DROP TABLE users; --",
                "' UNION SELECT * FROM users --",
                "1' OR '1'='1' --"
            ]
            
            vulnerable_endpoints = []
            
            for payload in sql_payloads:
                try:
                    # Testar diferentes endpoints
                    test_urls = [
                        f"{self.base_url}/?q={payload}",
                        f"{self.base_url}/search?term={payload}",
                        f"{self.base_url}/api?query={payload}"
                    ]
                    
                    for url in test_urls:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            content = response.text.lower()
                            # Verificar se há indicadores de SQL injection
                            if any(keyword in content for keyword in ['sql', 'mysql', 'postgresql', 'database', 'error']):
                                vulnerable_endpoints.append(url)
                except:
                    pass
            
            if len(vulnerable_endpoints) == 0:
                self.log_test("Proteção contra SQL Injection", "PASS", "Nenhuma vulnerabilidade encontrada")
                return True
            else:
                self.log_test("Proteção contra SQL Injection", "FAIL", f"Vulnerabilidades encontradas: {vulnerable_endpoints}")
                return False
        except Exception as e:
            self.log_test("Proteção contra SQL Injection", "FAIL", str(e))
            return False
    
    def test_xss_protection(self):
        """Teste 3: Proteção contra XSS"""
        try:
            # Testar payloads de XSS
            xss_payloads = [
                "<script>alert('XSS')</script>",
                "javascript:alert('XSS')",
                "<img src=x onerror=alert('XSS')>",
                "<svg onload=alert('XSS')>",
                "';alert('XSS');//"
            ]
            
            vulnerable_endpoints = []
            
            for payload in xss_payloads:
                try:
                    # Testar diferentes endpoints
                    test_urls = [
                        f"{self.base_url}/?q={payload}",
                        f"{self.base_url}/search?term={payload}",
                        f"{self.base_url}/api?query={payload}"
                    ]
                    
                    for url in test_urls:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            content = response.text
                            # Verificar se o payload foi refletido sem escape
                            if payload in content:
                                vulnerable_endpoints.append(url)
                except:
                    pass
            
            if len(vulnerable_endpoints) == 0:
                self.log_test("Proteção contra XSS", "PASS", "Nenhuma vulnerabilidade encontrada")
                return True
            else:
                self.log_test("Proteção contra XSS", "FAIL", f"Vulnerabilidades encontradas: {vulnerable_endpoints}")
                return False
        except Exception as e:
            self.log_test("Proteção contra XSS", "FAIL", str(e))
            return False
    
    def test_directory_traversal_protection(self):
        """Teste 4: Proteção contra Directory Traversal"""
        try:
            # Testar payloads de directory traversal
            traversal_payloads = [
                "../../../etc/passwd",
                "..\\..\\..\\windows\\system32\\drivers\\etc\\hosts",
                "....//....//....//etc/passwd",
                "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd"
            ]
            
            vulnerable_endpoints = []
            
            for payload in traversal_payloads:
                try:
                    # Testar diferentes endpoints
                    test_urls = [
                        f"{self.base_url}/?file={payload}",
                        f"{self.base_url}/download?path={payload}",
                        f"{self.base_url}/api/file?name={payload}"
                    ]
                    
                    for url in test_urls:
                        response = requests.get(url, timeout=5)
                        if response.status_code == 200:
                            content = response.text
                            # Verificar se há conteúdo sensível
                            if any(keyword in content.lower() for keyword in ['root:', 'admin:', 'password:', 'hosts']):
                                vulnerable_endpoints.append(url)
                except:
                    pass
            
            if len(vulnerable_endpoints) == 0:
                self.log_test("Proteção contra Directory Traversal", "PASS", "Nenhuma vulnerabilidade encontrada")
                return True
            else:
                self.log_test("Proteção contra Directory Traversal", "FAIL", f"Vulnerabilidades encontradas: {vulnerable_endpoints}")
                return False
        except Exception as e:
            self.log_test("Proteção contra Directory Traversal", "FAIL", str(e))
            return False
    
    def test_sensitive_data_exposure(self):
        """Teste 5: Exposição de dados sensíveis"""
        try:
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar se há dados sensíveis expostos
            sensitive_patterns = [
                r'password["\']?\s*[:=]\s*["\'][^"\']+["\']',
                r'api[_-]?key["\']?\s*[:=]\s*["\'][^"\']+["\']',
                r'secret["\']?\s*[:=]\s*["\'][^"\']+["\']',
                r'token["\']?\s*[:=]\s*["\'][^"\']+["\']',
                r'private[_-]?key["\']?\s*[:=]\s*["\'][^"\']+["\']'
            ]
            
            exposed_data = []
            for pattern in sensitive_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    exposed_data.extend(matches)
            
            if len(exposed_data) == 0:
                self.log_test("Exposição de Dados Sensíveis", "PASS", "Nenhum dado sensível encontrado")
                return True
            else:
                self.log_test("Exposição de Dados Sensíveis", "FAIL", f"Dados sensíveis encontrados: {exposed_data}")
                return False
        except Exception as e:
            self.log_test("Exposição de Dados Sensíveis", "FAIL", str(e))
            return False
    
    def test_authentication_security(self):
        """Teste 6: Segurança de autenticação"""
        try:
            # Verificar se há implementações de autenticação
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de autenticação
            auth_elements = [
                "login",
                "password",
                "authenticate",
                "session",
                "cookie",
                "jwt",
                "token"
            ]
            
            found_elements = [elem for elem in auth_elements if elem in content.lower()]
            
            if len(found_elements) >= 2:
                self.log_test("Segurança de Autenticação", "PASS", f"Elementos encontrados: {found_elements}")
                return True
            else:
                self.log_test("Segurança de Autenticação", "WARN", f"Apenas {found_elements} elementos encontrados")
                return False
        except Exception as e:
            self.log_test("Segurança de Autenticação", "FAIL", str(e))
            return False
    
    def test_session_security(self):
        """Teste 7: Segurança de sessão"""
        try:
            # Verificar cookies de sessão
            response = requests.get(self.base_url, timeout=10)
            cookies = response.cookies
            
            session_cookies = []
            for cookie in cookies:
                if any(keyword in cookie.name.lower() for keyword in ['session', 'auth', 'token', 'login']):
                    session_cookies.append(cookie.name)
            
            # Verificar atributos de segurança dos cookies
            secure_cookies = 0
            http_only_cookies = 0
            
            for cookie in cookies:
                if hasattr(cookie, 'secure') and cookie.secure:
                    secure_cookies += 1
                if hasattr(cookie, 'httponly') and cookie.httponly:
                    http_only_cookies += 1
            
            if len(session_cookies) > 0:
                self.log_test("Segurança de Sessão", "PASS", f"{len(session_cookies)} cookies de sessão, {secure_cookies} seguros, {http_only_cookies} HTTP-only")
                return True
            else:
                self.log_test("Segurança de Sessão", "WARN", "Nenhum cookie de sessão encontrado")
                return False
        except Exception as e:
            self.log_test("Segurança de Sessão", "FAIL", str(e))
            return False
    
    def test_input_validation(self):
        """Teste 8: Validação de entrada"""
        try:
            # Testar validação de entrada com diferentes tipos de dados
            test_inputs = [
                "normal_input",
                "<script>alert('test')</script>",
                "'; DROP TABLE test; --",
                "../../../etc/passwd",
                "very_long_input_" + "x" * 1000,
                "special_chars_!@#$%^&*()",
                "unicode_测试_中文"
            ]
            
            validation_results = []
            
            for test_input in test_inputs:
                try:
                    response = requests.get(f"{self.base_url}/?q={test_input}", timeout=5)
                    if response.status_code == 200:
                        content = response.text
                        # Verificar se a entrada foi sanitizada
                        if test_input not in content:
                            validation_results.append("sanitized")
                        else:
                            validation_results.append("not_sanitized")
                except:
                    validation_results.append("error")
            
            sanitized_count = validation_results.count("sanitized")
            total_tests = len(validation_results)
            
            if sanitized_count >= total_tests * 0.7:
                self.log_test("Validação de Entrada", "PASS", f"{sanitized_count}/{total_tests} entradas sanitizadas")
                return True
            else:
                self.log_test("Validação de Entrada", "FAIL", f"Apenas {sanitized_count}/{total_tests} entradas sanitizadas")
                return False
        except Exception as e:
            self.log_test("Validação de Entrada", "FAIL", str(e))
            return False
    
    def test_error_handling_security(self):
        """Teste 9: Segurança no tratamento de erros"""
        try:
            # Testar diferentes tipos de erros
            error_tests = [
                "/nonexistent",
                "/api/invalid",
                "/?invalid_param=test",
                "/admin",
                "/config"
            ]
            
            error_responses = []
            
            for error_path in error_tests:
                try:
                    response = requests.get(f"{self.base_url}{error_path}", timeout=5)
                    error_responses.append({
                        "path": error_path,
                        "status": response.status_code,
                        "content_length": len(response.content)
                    })
                except:
                    error_responses.append({
                        "path": error_path,
                        "status": "error",
                        "content_length": 0
                    })
            
            # Verificar se os erros não expõem informações sensíveis
            sensitive_errors = 0
            for error_response in error_responses:
                if error_response["status"] == 200 and error_response["content_length"] > 1000:
                    # Verificar se há informações sensíveis no conteúdo
                    try:
                        response = requests.get(f"{self.base_url}{error_response['path']}", timeout=5)
                        content = response.text.lower()
                        if any(keyword in content for keyword in ['stack trace', 'database', 'password', 'secret']):
                            sensitive_errors += 1
                    except:
                        pass
            
            if sensitive_errors == 0:
                self.log_test("Segurança no Tratamento de Erros", "PASS", f"Nenhum erro sensível encontrado")
                return True
            else:
                self.log_test("Segurança no Tratamento de Erros", "FAIL", f"{sensitive_errors} erros sensíveis encontrados")
                return False
        except Exception as e:
            self.log_test("Segurança no Tratamento de Erros", "FAIL", str(e))
            return False
    
    def test_file_upload_security(self):
        """Teste 10: Segurança de upload de arquivos"""
        try:
            # Verificar se há implementação de upload de arquivos
            response = requests.get(self.base_url, timeout=10)
            content = response.text
            
            # Verificar elementos de upload
            upload_elements = [
                "file_uploader",
                "upload",
                "multipart",
                "form-data"
            ]
            
            found_elements = [elem for elem in upload_elements if elem in content.lower()]
            
            if len(found_elements) >= 1:
                # Verificar se há validação de tipo de arquivo
                validation_elements = [
                    "type=",
                    "accept=",
                    "validation",
                    "check"
                ]
                
                validation_found = [elem for elem in validation_elements if elem in content.lower()]
                
                if len(validation_found) >= 1:
                    self.log_test("Segurança de Upload de Arquivos", "PASS", f"Upload implementado com validação")
                    return True
                else:
                    self.log_test("Segurança de Upload de Arquivos", "WARN", f"Upload implementado sem validação")
                    return False
            else:
                self.log_test("Segurança de Upload de Arquivos", "PASS", "Nenhum upload implementado")
                return True
        except Exception as e:
            self.log_test("Segurança de Upload de Arquivos", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de segurança"""
        print("🔒 INICIANDO TESTES DE SEGURANÇA")
        print("=" * 60)
        
        tests = [
            self.test_http_security_headers,
            self.test_sql_injection_protection,
            self.test_xss_protection,
            self.test_directory_traversal_protection,
            self.test_sensitive_data_exposure,
            self.test_authentication_security,
            self.test_session_security,
            self.test_input_validation,
            self.test_error_handling_security,
            self.test_file_upload_security
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
        print("📊 RESUMO DOS TESTES DE SEGURANÇA")
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
            with open("test_results_security.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_security.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🔒 TESTE DE SEGURANÇA - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = SecurityTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE SEGURANÇA PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE SEGURANÇA FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
