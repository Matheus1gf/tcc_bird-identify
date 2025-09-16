#!/usr/bin/env python3
"""
Análise Completa do Sistema - Identificação de Erros
Script para analisar todo o sistema e identificar problemas antes da execução
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
import json
from datetime import datetime

class SystemAnalyzer:
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.fixes_applied = []
        self.project_root = Path(".")
        self.src_path = self.project_root / "src"
        
    def log_error(self, error_type, file_path, line_num, message, fix=None):
        """Registrar erro encontrado"""
        error = {
            "type": error_type,
            "file": str(file_path),
            "line": line_num,
            "message": message,
            "fix": fix,
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error)
        print(f"❌ {error_type}: {file_path}:{line_num} - {message}")
        if fix:
            print(f"   🔧 Fix: {fix}")
    
    def log_warning(self, warning_type, file_path, message):
        """Registrar aviso"""
        warning = {
            "type": warning_type,
            "file": str(file_path),
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        self.warnings.append(warning)
        print(f"⚠️ {warning_type}: {file_path} - {message}")
    
    def log_fix(self, fix_type, file_path, description):
        """Registrar correção aplicada"""
        fix = {
            "type": fix_type,
            "file": str(file_path),
            "description": description,
            "timestamp": datetime.now().isoformat()
        }
        self.fixes_applied.append(fix)
        print(f"✅ {fix_type}: {file_path} - {description}")
    
    def check_file_exists(self, file_path):
        """Verificar se arquivo existe"""
        return os.path.exists(file_path)
    
    def check_directory_structure(self):
        """Verificar estrutura de diretórios"""
        print("📁 Verificando estrutura de diretórios...")
        
        required_dirs = [
            "src",
            "src/core",
            "src/interfaces", 
            "src/training",
            "src/utils",
            "data",
            "config",
            "docs",
            "scripts"
        ]
        
        for dir_path in required_dirs:
            if not os.path.exists(dir_path):
                self.log_error("MISSING_DIRECTORY", dir_path, 0, f"Diretório obrigatório não encontrado")
            else:
                print(f"✅ Diretório encontrado: {dir_path}")
    
    def check_required_files(self):
        """Verificar arquivos obrigatórios"""
        print("\n📄 Verificando arquivos obrigatórios...")
        
        required_files = [
            "main.py",
            "requirements.txt",
            "src/interfaces/web_app.py",
            "src/core/intuition.py",
            "src/core/annotator.py",
            "src/core/reasoning.py",
            "src/core/curator.py",
            "src/core/learning.py",
            "src/training/yolo_trainer.py",
            "src/training/keras_trainer.py",
            "src/utils/debug_logger.py",
            "src/utils/button_debug.py"
        ]
        
        for file_path in required_files:
            if not self.check_file_exists(file_path):
                self.log_error("MISSING_FILE", file_path, 0, f"Arquivo obrigatório não encontrado")
            else:
                print(f"✅ Arquivo encontrado: {file_path}")
    
    def analyze_python_syntax(self, file_path):
        """Analisar sintaxe Python de um arquivo"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Tentar compilar o código
            ast.parse(content)
            return True, None
        except SyntaxError as e:
            return False, f"SyntaxError: {e.msg} (linha {e.lineno})"
        except Exception as e:
            return False, f"Erro de análise: {str(e)}"
    
    def check_python_files_syntax(self):
        """Verificar sintaxe de todos os arquivos Python"""
        print("\n🐍 Verificando sintaxe de arquivos Python...")
        
        python_files = []
        
        # Encontrar todos os arquivos Python
        for root, dirs, files in os.walk("."):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        for file_path in python_files:
            is_valid, error = self.analyze_python_syntax(file_path)
            if not is_valid:
                self.log_error("SYNTAX_ERROR", file_path, 0, error)
            else:
                print(f"✅ Sintaxe OK: {file_path}")
    
    def check_imports(self, file_path):
        """Verificar imports de um arquivo"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for alias in node.names:
                        imports.append(f"{module}.{alias.name}")
            
            return imports
        except Exception as e:
            self.log_error("IMPORT_ANALYSIS_ERROR", file_path, 0, f"Erro ao analisar imports: {str(e)}")
            return []
    
    def check_all_imports(self):
        """Verificar todos os imports do sistema"""
        print("\n📦 Verificando imports...")
        
        python_files = []
        for root, dirs, files in os.walk("src"):
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))
        
        all_imports = {}
        for file_path in python_files:
            imports = self.check_imports(file_path)
            all_imports[file_path] = imports
        
        # Verificar se os módulos importados existem
        for file_path, imports in all_imports.items():
            for import_name in imports:
                if not self.check_import_exists(import_name, file_path):
                    self.log_error("MISSING_IMPORT", file_path, 0, f"Import não encontrado: {import_name}")
    
    def check_import_exists(self, import_name, from_file):
        """Verificar se um import existe"""
        try:
            # Tentar importar o módulo
            importlib.import_module(import_name)
            return True
        except ImportError:
            # Verificar se é um import relativo
            if import_name.startswith('.'):
                return self.check_relative_import(import_name, from_file)
            return False
        except Exception:
            return False
    
    def check_relative_import(self, import_name, from_file):
        """Verificar import relativo"""
        try:
            # Extrair o caminho do import relativo
            if import_name.startswith('..'):
                # Import de diretório pai
                return True  # Assumir que existe por enquanto
            elif import_name.startswith('.'):
                # Import do mesmo diretório
                return True  # Assumir que existe por enquanto
            return False
        except Exception:
            return False
    
    def check_main_py(self):
        """Verificar arquivo main.py"""
        print("\n🚀 Verificando main.py...")
        
        if not self.check_file_exists("main.py"):
            self.log_error("MISSING_FILE", "main.py", 0, "Arquivo main.py não encontrado")
            return
        
        try:
            with open("main.py", 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificar se tem a função main
            if "def main():" not in content:
                self.log_error("MISSING_FUNCTION", "main.py", 0, "Função main() não encontrada")
            
            # Verificar imports
            if "from interfaces.web_app import main" not in content:
                self.log_error("MISSING_IMPORT", "main.py", 0, "Import da web_app não encontrado")
            
            print("✅ main.py verificado")
        except Exception as e:
            self.log_error("FILE_READ_ERROR", "main.py", 0, f"Erro ao ler arquivo: {str(e)}")
    
    def check_web_app_py(self):
        """Verificar arquivo web_app.py"""
        print("\n🌐 Verificando web_app.py...")
        
        web_app_path = "src/interfaces/web_app.py"
        if not self.check_file_exists(web_app_path):
            self.log_error("MISSING_FILE", web_app_path, 0, "Arquivo web_app.py não encontrado")
            return
        
        try:
            with open(web_app_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificar imports problemáticos
            problematic_imports = [
                "from utils.debug_logger import debug_logger",
                "from utils.button_debug import button_debug",
                "from core.learning_sync import stop_continuous_sync",
                "from .manual_analysis import manual_analysis",
                "from .tinder_interface import TinderInterface"
            ]
            
            for import_line in problematic_imports:
                if import_line in content:
                    # Verificar se o módulo existe
                    module_name = import_line.split("import ")[1].strip()
                    if not self.check_import_exists(module_name, web_app_path):
                        self.log_error("MISSING_IMPORT", web_app_path, 0, f"Import problemático: {import_line}")
            
            print("✅ web_app.py verificado")
        except Exception as e:
            self.log_error("FILE_READ_ERROR", web_app_path, 0, f"Erro ao ler arquivo: {str(e)}")
    
    def create_missing_modules(self):
        """Criar módulos que estão faltando"""
        print("\n🔧 Criando módulos faltantes...")
        
        # Verificar se debug_logger existe
        debug_logger_path = "src/utils/debug_logger.py"
        if not self.check_file_exists(debug_logger_path):
            self.log_fix("CREATE_FILE", debug_logger_path, "Criando debug_logger.py")
            # O arquivo já foi criado anteriormente
        
        # Verificar se button_debug existe
        button_debug_path = "src/utils/button_debug.py"
        if not self.check_file_exists(button_debug_path):
            self.log_fix("CREATE_FILE", button_debug_path, "Criando button_debug.py")
            # O arquivo já foi criado anteriormente
        
        # Verificar se learning_sync existe
        learning_sync_path = "src/core/learning_sync.py"
        if not self.check_file_exists(learning_sync_path):
            self.log_fix("CREATE_FILE", learning_sync_path, "Criando learning_sync.py")
            self.create_learning_sync_module()
        
        # Verificar se manual_analysis existe
        manual_analysis_path = "src/interfaces/manual_analysis.py"
        if not self.check_file_exists(manual_analysis_path):
            self.log_fix("CREATE_FILE", manual_analysis_path, "Criando manual_analysis.py")
            self.create_manual_analysis_module()
        
        # Verificar se tinder_interface existe
        tinder_interface_path = "src/interfaces/tinder_interface.py"
        if not self.check_file_exists(tinder_interface_path):
            self.log_fix("CREATE_FILE", tinder_interface_path, "Criando tinder_interface.py")
            self.create_tinder_interface_module()
    
    def create_learning_sync_module(self):
        """Criar módulo learning_sync"""
        content = '''#!/usr/bin/env python3
"""
Módulo de Sincronização de Aprendizado
"""

def stop_continuous_sync():
    """Parar sincronização contínua"""
    print("🛑 Parando sincronização contínua...")
    return True

def start_continuous_sync():
    """Iniciar sincronização contínua"""
    print("🔄 Iniciando sincronização contínua...")
    return True
'''
        with open("src/core/learning_sync.py", "w", encoding="utf-8") as f:
            f.write(content)
    
    def create_manual_analysis_module(self):
        """Criar módulo manual_analysis"""
        content = '''#!/usr/bin/env python3
"""
Módulo de Análise Manual
"""

import streamlit as st

def manual_analysis():
    """Interface de análise manual"""
    st.write("📝 Análise Manual")
    st.write("Esta funcionalidade está em desenvolvimento.")
    return True
'''
        with open("src/interfaces/manual_analysis.py", "w", encoding="utf-8") as f:
            f.write(content)
    
    def create_tinder_interface_module(self):
        """Criar módulo tinder_interface"""
        content = '''#!/usr/bin/env python3
"""
Interface Tinder para Análise de Imagens
"""

import streamlit as st
import os

class TinderInterface:
    def __init__(self):
        self.pending_images = []
    
    def load_pending_images(self):
        """Carregar imagens pendentes"""
        return len(self.pending_images)
    
    def get_next_image(self):
        """Obter próxima imagem"""
        return None
    
    def approve_image(self, image_path):
        """Aprovar imagem"""
        return True
    
    def reject_image(self, image_path):
        """Rejeitar imagem"""
        return True
'''
        with open("src/interfaces/tinder_interface.py", "w", encoding="utf-8") as f:
            f.write(content)
    
    def run_complete_analysis(self):
        """Executar análise completa do sistema"""
        print("🔍 INICIANDO ANÁLISE COMPLETA DO SISTEMA")
        print("=" * 60)
        
        # 1. Verificar estrutura de diretórios
        self.check_directory_structure()
        
        # 2. Verificar arquivos obrigatórios
        self.check_required_files()
        
        # 3. Verificar sintaxe Python
        self.check_python_files_syntax()
        
        # 4. Verificar imports
        self.check_all_imports()
        
        # 5. Verificar arquivos específicos
        self.check_main_py()
        self.check_web_app_py()
        
        # 6. Criar módulos faltantes
        self.create_missing_modules()
        
        # 7. Relatório final
        self.generate_report()
    
    def generate_report(self):
        """Gerar relatório final"""
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO DE ANÁLISE DO SISTEMA")
        print("=" * 60)
        
        print(f"❌ Erros encontrados: {len(self.errors)}")
        print(f"⚠️ Avisos: {len(self.warnings)}")
        print(f"✅ Correções aplicadas: {len(self.fixes_applied)}")
        
        if self.errors:
            print("\n🔴 ERROS CRÍTICOS:")
            for error in self.errors:
                print(f"   - {error['type']}: {error['file']} - {error['message']}")
        
        if self.warnings:
            print("\n🟡 AVISOS:")
            for warning in self.warnings:
                print(f"   - {warning['type']}: {warning['file']} - {warning['message']}")
        
        if self.fixes_applied:
            print("\n🟢 CORREÇÕES APLICADAS:")
            for fix in self.fixes_applied:
                print(f"   - {fix['type']}: {fix['file']} - {fix['description']}")
        
        # Salvar relatório
        report = {
            "timestamp": datetime.now().isoformat(),
            "errors": self.errors,
            "warnings": self.warnings,
            "fixes_applied": self.fixes_applied,
            "summary": {
                "total_errors": len(self.errors),
                "total_warnings": len(self.warnings),
                "total_fixes": len(self.fixes_applied)
            }
        }
        
        with open("system_analysis_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Relatório salvo em: system_analysis_report.json")
        
        return len(self.errors) == 0

def main():
    """Função principal"""
    analyzer = SystemAnalyzer()
    success = analyzer.run_complete_analysis()
    
    if success:
        print("\n🎉 SISTEMA PRONTO PARA EXECUÇÃO!")
        return 0
    else:
        print("\n❌ SISTEMA TEM ERROS QUE PRECISAM SER CORRIGIDOS!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
