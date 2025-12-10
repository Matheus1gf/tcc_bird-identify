#!/usr/bin/env python3
"""
Launcher Universal - Sistema de Identificação de Pássaros
Detecta sistema operacional, instala dependências e inicia a aplicação
"""

import os
import sys
import platform
import subprocess
import importlib
import time
import json
from pathlib import Path
from typing import Dict, List, Optional

class UniversalLauncher:
    """
    Launcher universal que detecta SO, instala dependências e inicia aplicação
    """
    
    def __init__(self):
        self.system_info = self._detect_system()
        self.dependencies = self._load_dependencies()
        self.installation_log = []
        
    def _detect_system(self) -> Dict:
        """Detecta informações do sistema operacional"""
        system_info = {
            "os": platform.system().lower(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "architecture": platform.machine(),
            "platform": platform.platform(),
            "python_executable": sys.executable
        }
        
        print("🔍 Detectando sistema operacional...")
        print(f"  • SO: {system_info['os']}")
        print(f"  • Versão: {system_info['os_version']}")
        print(f"  • Python: {system_info['python_version']}")
        print(f"  • Arquitetura: {system_info['architecture']}")
        
        return system_info
    
    def _load_dependencies(self) -> List[Dict]:
        """Carrega lista de dependências do requirements.txt"""
        dependencies = []
        
        try:
            with open("requirements.txt", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        # Parse da dependência
                        if "==" in line:
                            name, version = line.split("==")
                            dependencies.append({
                                "name": name,
                                "version": version,
                                "required": True
                            })
                        else:
                            dependencies.append({
                                "name": line,
                                "version": None,
                                "required": True
                            })
        except FileNotFoundError:
            print("⚠️ Arquivo requirements.txt não encontrado")
            # Dependências básicas como fallback
            dependencies = [
                {"name": "streamlit", "version": "1.28.0", "required": True},
                {"name": "numpy", "version": "1.24.3", "required": True},
                {"name": "opencv-python", "version": "4.8.1.78", "required": True},
                {"name": "pillow", "version": "10.0.0", "required": True},
                {"name": "tensorflow", "version": "2.13.0", "required": True},
                {"name": "ultralytics", "version": "8.0.196", "required": True},
                {"name": "matplotlib", "version": "3.7.2", "required": True},
                {"name": "pandas", "version": "2.0.3", "required": True},
                {"name": "scikit-learn", "version": "1.3.0", "required": True},
                {"name": "scikit-image", "version": "0.21.0", "required": True},
                {"name": "networkx", "version": "3.1", "required": True},
                {"name": "requests", "version": "2.31.0", "required": True},
                {"name": "plotly", "version": "5.17.0", "required": True}
            ]
        
        print(f"📦 {len(dependencies)} dependências carregadas")
        return dependencies
    
    def check_dependencies(self) -> Dict:
        """Verifica quais dependências estão instaladas"""
        print("\n🔍 Verificando dependências instaladas...")
        
        installed = []
        missing = []
        
        for dep in self.dependencies:
            try:
                # Tentar importar o módulo
                if dep["name"] == "opencv-python":
                    importlib.import_module("cv2")
                elif dep["name"] == "scikit-image":
                    importlib.import_module("skimage")
                elif dep["name"] == "scikit-learn":
                    importlib.import_module("sklearn")
                elif dep["name"] == "google-generativeai":
                    importlib.import_module("google.generativeai")
                elif dep["name"] == "setuptools>=65.0.0":
                    importlib.import_module("setuptools")
                elif dep["name"] == "wheel>=0.37.0":
                    importlib.import_module("wheel")
                elif dep["name"] == "pillow":
                    importlib.import_module("PIL")
                else:
                    importlib.import_module(dep["name"].replace("-", "_"))
                
                installed.append(dep)
                print(f"  ✅ {dep['name']} - OK")
                
            except ImportError:
                missing.append(dep)
                print(f"  ❌ {dep['name']} - FALTANDO")
        
        return {
            "installed": installed,
            "missing": missing,
            "total_installed": len(installed),
            "total_missing": len(missing)
        }
    
    def install_dependencies(self, missing_deps: List[Dict]) -> bool:
        """Instala dependências faltantes"""
        if not missing_deps:
            print("✅ Todas as dependências já estão instaladas!")
            return True
        
        print(f"\n📦 Instalando {len(missing_deps)} dependências faltantes...")
        
        # Preparar comando de instalação baseado no SO
        if self.system_info["os"] == "windows":
            pip_cmd = [sys.executable, "-m", "pip", "install"]
        else:
            pip_cmd = ["python3", "-m", "pip", "install"]
        
        # Instalar cada dependência
        for dep in missing_deps:
            try:
                print(f"  🔄 Instalando {dep['name']}...")
                
                if dep["version"]:
                    # Remover >= da versão para instalação
                    clean_name = dep["name"].split(">=")[0]
                    package = f"{clean_name}=={dep['version']}"
                else:
                    package = dep["name"]
                
                cmd = pip_cmd + [package, "--upgrade"]
                
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True, 
                    timeout=300  # 5 minutos timeout
                )
                
                if result.returncode == 0:
                    print(f"  ✅ {dep['name']} instalado com sucesso")
                    self.installation_log.append({
                        "package": dep["name"],
                        "status": "success",
                        "output": result.stdout
                    })
                else:
                    print(f"  ❌ Erro ao instalar {dep['name']}: {result.stderr}")
                    self.installation_log.append({
                        "package": dep["name"],
                        "status": "error",
                        "output": result.stderr
                    })
                    
            except subprocess.TimeoutExpired:
                print(f"  ⏰ Timeout ao instalar {dep['name']}")
                self.installation_log.append({
                    "package": dep["name"],
                    "status": "timeout",
                    "output": "Timeout"
                })
            except Exception as e:
                print(f"  ❌ Erro inesperado ao instalar {dep['name']}: {e}")
                self.installation_log.append({
                    "package": dep["name"],
                    "status": "error",
                    "output": str(e)
                })
        
        # Verificar se todas foram instaladas
        final_check = self.check_dependencies()
        return final_check["total_missing"] == 0
    
    def setup_environment(self):
        """Configura ambiente necessário"""
        print("\n🔧 Configurando ambiente...")
        
        # Criar diretórios necessários
        directories = [
            "manual_analysis/pending",
            "manual_analysis/approved", 
            "manual_analysis/rejected",
            "manual_analysis/annotations",
            "learning_data/auto_approved",
            "learning_data/auto_rejected",
            "learning_data/awaiting_human_review",
            "learning_data/cycles_history",
            "learning_data/model_checkpoints",
            "learning_data/pending_validation",
            "dataset_passaros/images/train",
            "dataset_passaros/images/val",
            "dataset_passaros/labels",
            "runs/detect",
            "runs/train"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            print(f"  📁 Diretório criado/verificado: {directory}")
        
        # Verificar se modelos existem
        self._check_models()
        
        print("✅ Ambiente configurado com sucesso!")
    
    def _check_models(self):
        """Verifica se modelos necessários existem"""
        print("\n🤖 Verificando modelos...")
        
        models_to_check = [
            ("yolov8n.pt", "Modelo YOLO"),
            ("modelo_classificacao_passaros.keras", "Modelo Keras"),
            ("modelo_classificacao_passaros.h5", "Modelo Keras H5")
        ]
        
        for model_file, description in models_to_check:
            if os.path.exists(model_file):
                print(f"  ✅ {description}: {model_file}")
            else:
                print(f"  ⚠️ {description}: {model_file} - NÃO ENCONTRADO")
                if model_file == "yolov8n.pt":
                    print("    💡 O modelo YOLO será baixado automaticamente na primeira execução")
    
    def start_application(self):
        """Inicia a aplicação principal"""
        print("\n🚀 Iniciando aplicação...")
        
        try:
            # Verificar se app.py existe
            if not os.path.exists("app.py"):
                print("❌ Arquivo app.py não encontrado!")
                return False
            
            # Determinar comando baseado no SO
            if self.system_info["os"] == "windows":
                cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
            else:
                cmd = ["python3", "-m", "streamlit", "run", "app.py", "--server.port", "8501"]
            
            print(f"  🌐 Iniciando Streamlit na porta 8501...")
            print(f"  📱 Acesse: http://localhost:8501")
            print(f"  🔄 Comando: {' '.join(cmd)}")
            print("\n" + "="*60)
            print("🎉 APLICAÇÃO INICIADA COM SUCESSO!")
            print("="*60)
            
            # Executar aplicação
            subprocess.run(cmd)
            
        except KeyboardInterrupt:
            print("\n⏹️ Aplicação interrompida pelo usuário")
        except Exception as e:
            print(f"\n❌ Erro ao iniciar aplicação: {e}")
            return False
        
        return True
    
    def run_full_setup(self):
        """Executa configuração completa e inicia aplicação"""
        print("🎯 LAUNCHER UNIVERSAL - Sistema de Identificação de Pássaros")
        print("=" * 70)
        
        try:
            # 1. Verificar dependências
            deps_status = self.check_dependencies()
            
            # 2. Instalar dependências faltantes
            if deps_status["total_missing"] > 0:
                print(f"\n📦 Instalando {deps_status['total_missing']} dependências faltantes...")
                install_success = self.install_dependencies(deps_status["missing"])
                
                if not install_success:
                    print("❌ Falha na instalação de dependências!")
                    return False
            else:
                print("✅ Todas as dependências já estão instaladas!")
            
            # 3. Configurar ambiente
            self.setup_environment()
            
            # 4. Iniciar sincronização contínua
            print("\n🔄 Iniciando sistema de sincronização...")
            try:
                from learning_sync import start_continuous_sync
                start_continuous_sync()
                print("✅ Sistema de sincronização iniciado")
            except ImportError:
                print("⚠️ Módulo de sincronização não encontrado")
            
            # 5. Iniciar aplicação
            return self.start_application()
            
        except Exception as e:
            print(f"\n❌ Erro durante configuração: {e}")
            return False
    
    def save_installation_log(self):
        """Salva log de instalação"""
        log_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": self.system_info,
            "installation_log": self.installation_log
        }
        
        with open("installation_log.json", "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
        
        print(f"📋 Log de instalação salvo em: installation_log.json")

def main():
    """Função principal"""
    launcher = UniversalLauncher()
    
    try:
        success = launcher.run_full_setup()
        
        if success:
            print("\n🎉 Sistema configurado e iniciado com sucesso!")
        else:
            print("\n❌ Falha na configuração do sistema")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⏹️ Instalação interrompida pelo usuário")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        sys.exit(1)
    finally:
        launcher.save_installation_log()

if __name__ == "__main__":
    main()
