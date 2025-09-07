#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface de Linha de Comando (CLI) para Análise de Pássaros
Funciona em qualquer sistema operacional
"""

import os
import sys
import glob
from datetime import datetime

# Adicionar o diretório atual ao sys.path para importar módulos
sys.path.append('.')

try:
    from logical_ai_reasoning_system import LogicalAIReasoningSystem
    from manual_analysis_system import ManualAnalysisSystem
    from button_debug_logger import ButtonDebugLogger
except ImportError as e:
    print(f"❌ Erro ao importar módulos: {e}")
    sys.exit(1)

class BirdAnalysisCLI:
    """Interface CLI para análise de pássaros"""
    
    def __init__(self):
        self.system = None
        self.manual_analysis = ManualAnalysisSystem()
        self.button_debug = ButtonDebugLogger("cli_button_debug.log")
        self.current_image_path = None
        self.temp_path = None
        
        print("🐦 Sistema de Análise de Pássaros - CLI")
        print("=" * 50)
        
        # Inicializar sistema
        self.init_system()
    
    def init_system(self):
        """Inicializar sistema"""
        try:
            print("🔄 Inicializando sistema...")
            self.system = LogicalAIReasoningSystem()
            print("✅ Sistema inicializado com sucesso!")
        except Exception as e:
            print(f"❌ Erro ao inicializar sistema: {e}")
            sys.exit(1)
    
    def show_menu(self):
        """Mostrar menu principal"""
        print("\n" + "=" * 50)
        print("📋 MENU PRINCIPAL")
        print("=" * 50)
        print("1. 📁 Selecionar imagem")
        print("2. 🔍 Analisar imagem")
        print("3. 📋 Marcar para análise manual")
        print("4. 🧪 Teste de botão")
        print("5. 📊 Ver imagens pendentes")
        print("6. 🗑️ Limpar arquivos temporários")
        print("7. ❌ Sair")
        print("=" * 50)
    
    def select_image(self):
        """Selecionar imagem"""
        print("\n📁 SELECIONAR IMAGEM")
        print("-" * 30)
        
        # Procurar imagens no diretório atual
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
        images = []
        
        for ext in image_extensions:
            images.extend(glob.glob(ext))
            images.extend(glob.glob(ext.upper()))
        
        if not images:
            print("❌ Nenhuma imagem encontrada no diretório atual!")
            print("💡 Coloque uma imagem (.jpg, .png, etc.) neste diretório e tente novamente.")
            return False
        
        print("📸 Imagens encontradas:")
        for i, img in enumerate(images, 1):
            print(f"  {i}. {img}")
        
        try:
            choice = int(input("\nEscolha uma imagem (número): ")) - 1
            if 0 <= choice < len(images):
                self.current_image_path = images[choice]
                print(f"✅ Imagem selecionada: {self.current_image_path}")
                return True
            else:
                print("❌ Escolha inválida!")
                return False
        except ValueError:
            print("❌ Digite um número válido!")
            return False
    
    def analyze_image(self):
        """Analisar imagem"""
        if not self.current_image_path:
            print("❌ Nenhuma imagem selecionada!")
            return False
        
        print(f"\n🔍 ANALISANDO IMAGEM: {self.current_image_path}")
        print("-" * 50)
        
        try:
            # Copiar imagem para arquivo temporário
            self.temp_path = f"temp_{os.path.basename(self.current_image_path)}"
            
            # Usar PIL para copiar a imagem
            from PIL import Image
            with Image.open(self.current_image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(self.temp_path, format='PNG')
            
            print(f"📸 Imagem temporária criada: {self.temp_path}")
            
            # Analisar com o sistema
            print("🔄 Executando análise...")
            result = self.system.analyze_image_revolutionary(self.temp_path)
            
            # Exibir resultados
            self.display_results(result)
            
            print("✅ Análise concluída!")
            return True
            
        except Exception as e:
            print(f"❌ Erro na análise: {e}")
            return False
    
    def display_results(self, result):
        """Exibir resultados da análise"""
        print("\n📊 RESULTADOS DA ANÁLISE")
        print("=" * 50)
        
        # Informações básicas
        print(f"📁 Arquivo: {result.get('image_path', 'N/A')}")
        print(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")
        
        # Análise YOLO
        if 'intuition_analysis' in result and 'yolo_analysis' in result['intuition_analysis']:
            yolo_data = result['intuition_analysis']['yolo_analysis']
            print(f"🔍 Detecções YOLO: {yolo_data.get('total_detections', 0)}")
            
            for i, detection in enumerate(yolo_data.get('detections', [])):
                print(f"  {i+1}. {detection.get('class', 'N/A')} - {detection.get('confidence', 0):.2%}")
        
        # Recomendação
        if 'intuition_analysis' in result and 'intuition_analysis' in result['intuition_analysis']:
            intuition_data = result['intuition_analysis']['intuition_analysis']
            print(f"🧠 Nível de intuição: {intuition_data.get('intuition_level', 'N/A')}")
            print(f"💡 Recomendação: {intuition_data.get('recommendation', 'N/A')}")
        
        print("=" * 50)
    
    def mark_for_manual_analysis(self):
        """Marcar imagem para análise manual"""
        if not self.temp_path:
            print("❌ Nenhuma imagem analisada!")
            return False
        
        print(f"\n📋 MARCANDO PARA ANÁLISE MANUAL")
        print("-" * 40)
        
        try:
            # LOG DETALHADO DO BOTÃO
            self.button_debug.log_button_click("cli_manual_analysis", self.temp_path)
            self.button_debug.log_step("INICIANDO PROCESSO DE ANÁLISE MANUAL")
            
            # Verificar se arquivo temporário existe
            file_exists = os.path.exists(self.temp_path)
            self.button_debug.log_file_check(self.temp_path, file_exists)
            
            if not file_exists:
                error_msg = f"Arquivo temporário não encontrado: {self.temp_path}"
                print(f"❌ {error_msg}")
                self.button_debug.log_error(error_msg, "FILE_NOT_FOUND")
                return False
            else:
                self.button_debug.log_success("Arquivo temporário encontrado")
                
                # Adicionar à fila de análise manual
                detection_data = {
                    'yolo_detections': [],
                    'confidence': 0.0,
                    'analysis_type': 'cli_manual',
                    'timestamp': datetime.now().isoformat()
                }
                
                self.button_debug.log_step("PREPARANDO DADOS DE DETECÇÃO", str(detection_data))
                
                self.button_debug.log_manual_analysis_call(self.temp_path, detection_data)
                pending_path = self.manual_analysis.add_image_for_analysis(self.temp_path, detection_data)
                
                # Verificar se foi criada
                success = os.path.exists(pending_path)
                self.button_debug.log_manual_analysis_result(pending_path, success)
                
                if success:
                    self.button_debug.log_success("Arquivo copiado com sucesso!")
                    print("✅ Imagem marcada para análise manual!")
                    print(f"📁 Salva em: {pending_path}")
                else:
                    self.button_debug.log_error("Falha ao copiar arquivo!", "COPY_ERROR")
                    print("❌ Falha ao copiar arquivo!")
                    return False
                
                # Remover arquivo temporário após copiar
                if os.path.exists(self.temp_path):
                    os.remove(self.temp_path)
                    self.button_debug.log_file_cleanup(self.temp_path, True)
                    print("🗑️ Arquivo temporário removido")
                else:
                    self.button_debug.log_file_cleanup(self.temp_path, False)
                
                return True
            
        except Exception as e:
            error_msg = f"Erro ao adicionar para análise manual: {e}"
            print(f"❌ {error_msg}")
            self.button_debug.log_error(error_msg, "MANUAL_ANALYSIS_ERROR")
            return False
        finally:
            self.button_debug.log_session_end()
    
    def test_button_function(self):
        """Teste de botão simples"""
        print("\n🧪 TESTE DE BOTÃO")
        print("-" * 20)
        print("✅ Botão funcionando perfeitamente!")
        print("✅ Sistema CLI funcionando!")
        return True
    
    def show_pending_images(self):
        """Mostrar imagens pendentes"""
        print("\n📊 IMAGENS PENDENTES")
        print("-" * 30)
        
        pending_dir = self.manual_analysis.pending_dir
        if os.path.exists(pending_dir):
            files = os.listdir(pending_dir)
            image_files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))]
            
            if image_files:
                print(f"📁 Encontradas {len(image_files)} imagens:")
                for img in image_files:
                    file_path = os.path.join(pending_dir, img)
                    file_size = os.path.getsize(file_path)
                    print(f"  • {img} ({file_size} bytes)")
            else:
                print("📭 Nenhuma imagem pendente")
        else:
            print("❌ Diretório de imagens pendentes não encontrado")
    
    def clear_temp_files(self):
        """Limpar arquivos temporários"""
        print("\n🗑️ LIMPANDO ARQUIVOS TEMPORÁRIOS")
        print("-" * 40)
        
        temp_files = glob.glob("temp_*")
        if temp_files:
            for file in temp_files:
                try:
                    os.remove(file)
                    print(f"🗑️ Removido: {file}")
                except Exception as e:
                    print(f"❌ Erro ao remover {file}: {e}")
            print(f"✅ {len(temp_files)} arquivos temporários removidos")
        else:
            print("📭 Nenhum arquivo temporário encontrado")
    
    def run(self):
        """Executar aplicação CLI"""
        while True:
            self.show_menu()
            
            try:
                choice = input("\nEscolha uma opção (1-7): ").strip()
                
                if choice == "1":
                    self.select_image()
                elif choice == "2":
                    self.analyze_image()
                elif choice == "3":
                    self.mark_for_manual_analysis()
                elif choice == "4":
                    self.test_button_function()
                elif choice == "5":
                    self.show_pending_images()
                elif choice == "6":
                    self.clear_temp_files()
                elif choice == "7":
                    print("\n👋 Saindo do sistema...")
                    break
                else:
                    print("❌ Opção inválida! Escolha de 1 a 7.")
                
                input("\n⏸️ Pressione Enter para continuar...")
                
            except KeyboardInterrupt:
                print("\n\n👋 Saindo do sistema...")
                break
            except Exception as e:
                print(f"\n❌ Erro: {e}")
                input("⏸️ Pressione Enter para continuar...")

def main():
    """Função principal"""
    app = BirdAnalysisCLI()
    app.run()

if __name__ == "__main__":
    main()
