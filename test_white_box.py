#!/usr/bin/env python3
"""
Teste de Caixa Branca - Sistema de Identificação de Pássaros
Testa código interno com conhecimento da implementação
"""

import ast
import os
import sys
import json
from datetime import datetime
import importlib.util

class WhiteBoxTester:
    def __init__(self):
        self.results = []
        self.source_files = [
            "src/interfaces/web_app.py",
            "src/core/intuition.py",
            "src/core/reasoning.py",
            "src/core/learning.py",
            "src/core/cache.py",
            "src/utils/debug_logger.py",
            "src/utils/button_debug.py"
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
    
    def test_syntax_validation(self):
        """Teste 1: Validação de sintaxe"""
        try:
            syntax_errors = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    try:
                        with open(file_path, 'r') as f:
                            content = f.read()
                        ast.parse(content)
                    except SyntaxError as e:
                        syntax_errors.append(f"{file_path}:{e.lineno} - {e.msg}")
                    except IndentationError as e:
                        syntax_errors.append(f"{file_path}:{e.lineno} - {e.msg}")
            
            if len(syntax_errors) == 0:
                self.log_test("Validação de Sintaxe", "PASS", f"{len(self.source_files)} arquivos validados")
                return True
            else:
                self.log_test("Validação de Sintaxe", "FAIL", f"Erros encontrados: {syntax_errors}")
                return False
        except Exception as e:
            self.log_test("Validação de Sintaxe", "FAIL", str(e))
            return False
    
    def test_import_structure(self):
        """Teste 2: Estrutura de imports"""
        try:
            import_issues = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if stripped.startswith('import ') or stripped.startswith('from '):
                            # Verificar se o import é válido
                            try:
                                compile(stripped, '<string>', 'exec')
                            except SyntaxError:
                                import_issues.append(f"{file_path}:{i} - Import inválido: {stripped}")
            
            if len(import_issues) == 0:
                self.log_test("Estrutura de Imports", "PASS", f"{len(self.source_files)} arquivos verificados")
                return True
            else:
                self.log_test("Estrutura de Imports", "FAIL", f"Issues encontradas: {import_issues}")
                return False
        except Exception as e:
            self.log_test("Estrutura de Imports", "FAIL", str(e))
            return False
    
    def test_function_coverage(self):
        """Teste 3: Cobertura de funções"""
        try:
            function_stats = {}
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    
                    function_stats[file_path] = {
                        "functions": len(functions),
                        "classes": len(classes),
                        "function_names": functions,
                        "class_names": classes
                    }
            
            total_functions = sum(stats["functions"] for stats in function_stats.values())
            total_classes = sum(stats["classes"] for stats in function_stats.values())
            
            if total_functions > 0 and total_classes > 0:
                self.log_test("Cobertura de Funções", "PASS", f"{total_functions} funções, {total_classes} classes")
                return True
            else:
                self.log_test("Cobertura de Funções", "FAIL", f"{total_functions} funções, {total_classes} classes")
                return False
        except Exception as e:
            self.log_test("Cobertura de Funções", "FAIL", str(e))
            return False
    
    def test_error_handling(self):
        """Teste 4: Tratamento de erros"""
        try:
            error_handling_stats = {}
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Contar blocos try/except
                    try_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.Try)])
                    except_blocks = len([node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)])
                    
                    error_handling_stats[file_path] = {
                        "try_blocks": try_blocks,
                        "except_blocks": except_blocks
                    }
            
            total_try = sum(stats["try_blocks"] for stats in error_handling_stats.values())
            total_except = sum(stats["except_blocks"] for stats in error_handling_stats.values())
            
            if total_try > 0 and total_except > 0:
                self.log_test("Tratamento de Erros", "PASS", f"{total_try} try blocks, {total_except} except blocks")
                return True
            else:
                self.log_test("Tratamento de Erros", "WARN", f"{total_try} try blocks, {total_except} except blocks")
                return False
        except Exception as e:
            self.log_test("Tratamento de Erros", "FAIL", str(e))
            return False
    
    def test_code_complexity(self):
        """Teste 5: Complexidade do código"""
        try:
            complexity_stats = {}
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Contar estruturas de controle
                    if_statements = len([node for node in ast.walk(tree) if isinstance(node, ast.If)])
                    for_loops = len([node for node in ast.walk(tree) if isinstance(node, ast.For)])
                    while_loops = len([node for node in ast.walk(tree) if isinstance(node, ast.While)])
                    function_defs = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                    
                    complexity_stats[file_path] = {
                        "if_statements": if_statements,
                        "for_loops": for_loops,
                        "while_loops": while_loops,
                        "function_defs": function_defs,
                        "total_lines": len(content.split('\n'))
                    }
            
            total_complexity = sum(
                stats["if_statements"] + stats["for_loops"] + stats["while_loops"] 
                for stats in complexity_stats.values()
            )
            
            if total_complexity > 0:
                self.log_test("Complexidade do Código", "PASS", f"Complexidade total: {total_complexity}")
                return True
            else:
                self.log_test("Complexidade do Código", "FAIL", f"Complexidade total: {total_complexity}")
                return False
        except Exception as e:
            self.log_test("Complexidade do Código", "FAIL", str(e))
            return False
    
    def test_documentation_coverage(self):
        """Teste 6: Cobertura de documentação"""
        try:
            doc_stats = {}
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Contar docstrings
                    docstrings = 0
                    for node in ast.walk(tree):
                        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Module)):
                            if ast.get_docstring(node):
                                docstrings += 1
                    
                    # Contar comentários
                    comments = len([line for line in content.split('\n') if line.strip().startswith('#')])
                    
                    doc_stats[file_path] = {
                        "docstrings": docstrings,
                        "comments": comments,
                        "total_lines": len(content.split('\n'))
                    }
            
            total_docs = sum(stats["docstrings"] for stats in doc_stats.values())
            total_comments = sum(stats["comments"] for stats in doc_stats.values())
            
            if total_docs > 0 or total_comments > 0:
                self.log_test("Cobertura de Documentação", "PASS", f"{total_docs} docstrings, {total_comments} comentários")
                return True
            else:
                self.log_test("Cobertura de Documentação", "WARN", f"{total_docs} docstrings, {total_comments} comentários")
                return False
        except Exception as e:
            self.log_test("Cobertura de Documentação", "FAIL", str(e))
            return False
    
    def test_security_patterns(self):
        """Teste 7: Padrões de segurança"""
        try:
            security_issues = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Verificar padrões inseguros
                    insecure_patterns = [
                        "eval(",
                        "exec(",
                        "os.system(",
                        "subprocess.call(",
                        "pickle.loads(",
                        "yaml.load(",
                        "json.loads(",
                        "input("
                    ]
                    
                    for pattern in insecure_patterns:
                        if pattern in content:
                            security_issues.append(f"{file_path} - Padrão inseguro: {pattern}")
            
            if len(security_issues) == 0:
                self.log_test("Padrões de Segurança", "PASS", "Nenhum padrão inseguro encontrado")
                return True
            else:
                self.log_test("Padrões de Segurança", "WARN", f"Padrões inseguros: {security_issues}")
                return False
        except Exception as e:
            self.log_test("Padrões de Segurança", "FAIL", str(e))
            return False
    
    def test_code_standards(self):
        """Teste 8: Padrões de código"""
        try:
            standards_issues = []
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    for i, line in enumerate(lines, 1):
                        # Verificar indentação (deve ser múltiplo de 4)
                        if line.strip() and not line.startswith('#'):
                            stripped = line.lstrip()
                            if stripped:
                                indent = len(line) - len(stripped)
                                if indent > 0 and indent % 4 != 0:
                                    standards_issues.append(f"{file_path}:{i} - Indentação inconsistente")
                        
                        # Verificar linhas muito longas (>100 caracteres)
                        if len(line.rstrip()) > 100:
                            standards_issues.append(f"{file_path}:{i} - Linha muito longa")
            
            if len(standards_issues) == 0:
                self.log_test("Padrões de Código", "PASS", "Código segue padrões")
                return True
            else:
                self.log_test("Padrões de Código", "WARN", f"Issues encontradas: {len(standards_issues)}")
                return False
        except Exception as e:
            self.log_test("Padrões de Código", "FAIL", str(e))
            return False
    
    def test_dependency_analysis(self):
        """Teste 9: Análise de dependências"""
        try:
            dependencies = set()
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    # Extrair imports
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                dependencies.add(alias.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.module:
                                dependencies.add(node.module)
            
            external_deps = [dep for dep in dependencies if not dep.startswith('src.')]
            
            if len(external_deps) > 0:
                self.log_test("Análise de Dependências", "PASS", f"{len(external_deps)} dependências externas")
                return True
            else:
                self.log_test("Análise de Dependências", "FAIL", "Nenhuma dependência externa encontrada")
                return False
        except Exception as e:
            self.log_test("Análise de Dependências", "FAIL", str(e))
            return False
    
    def test_code_metrics(self):
        """Teste 10: Métricas de código"""
        try:
            total_lines = 0
            total_functions = 0
            total_classes = 0
            
            for file_path in self.source_files:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    tree = ast.parse(content)
                    
                    total_lines += len(content.split('\n'))
                    total_functions += len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])
                    total_classes += len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
            
            if total_lines > 0 and total_functions > 0:
                self.log_test("Métricas de Código", "PASS", f"{total_lines} linhas, {total_functions} funções, {total_classes} classes")
                return True
            else:
                self.log_test("Métricas de Código", "FAIL", f"{total_lines} linhas, {total_functions} funções, {total_classes} classes")
                return False
        except Exception as e:
            self.log_test("Métricas de Código", "FAIL", str(e))
            return False
    
    def run_all_tests(self):
        """Executar todos os testes de caixa branca"""
        print("⚪ INICIANDO TESTES DE CAIXA BRANCA")
        print("=" * 60)
        
        tests = [
            self.test_syntax_validation,
            self.test_import_structure,
            self.test_function_coverage,
            self.test_error_handling,
            self.test_code_complexity,
            self.test_documentation_coverage,
            self.test_security_patterns,
            self.test_code_standards,
            self.test_dependency_analysis,
            self.test_code_metrics
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
        
        # Resumo
        print("\n" + "=" * 60)
        print("📊 RESUMO DOS TESTES DE CAIXA BRANCA")
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
            with open("test_results_white_box.json", "w") as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            print(f"\n💾 Resultados salvos em: test_results_white_box.json")
        except Exception as e:
            print(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("⚪ TESTE DE CAIXA BRANCA - SISTEMA DE IDENTIFICAÇÃO DE PÁSSAROS")
    print("=" * 70)
    
    tester = WhiteBoxTester()
    
    try:
        passed, failed, warned = tester.run_all_tests()
        
        if failed == 0:
            print("\n🎉 TODOS OS TESTES DE CAIXA BRANCA PASSARAM!")
            sys.exit(0)
        else:
            print(f"\n⚠️ {failed} TESTES DE CAIXA BRANCA FALHARAM")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Testes interrompidos pelo usuário")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
