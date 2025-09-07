#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Interface Desktop para Análise de Pássaros
Usando Tkinter (framework nativo do Python)
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
from PIL import Image, ImageTk
import threading
from datetime import datetime

# Adicionar o diretório atual ao sys.path para importar módulos
sys.path.append('.')

try:
    from logical_ai_reasoning_system import LogicalAIReasoningSystem
    from manual_analysis_system import ManualAnalysisSystem
    from button_debug_logger import ButtonDebugLogger
except ImportError as e:
    print(f"Erro ao importar módulos: {e}")
    sys.exit(1)

class BirdAnalysisDesktopApp:
    """Interface desktop para análise de pássaros"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análise de Pássaros - Desktop")
        self.root.geometry("1200x800")
        
        # Inicializar sistemas
        self.system = None
        self.manual_analysis = ManualAnalysisSystem()
        self.button_debug = ButtonDebugLogger("desktop_button_debug.log")
        
        # Variáveis
        self.current_image_path = None
        self.temp_path = None
        
        # Criar interface
        self.create_widgets()
        
        # Inicializar sistema em thread separada
        self.init_system()
    
    def create_widgets(self):
        """Criar widgets da interface"""
        # Frame principal
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configurar grid
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Título
        title_label = ttk.Label(main_frame, text="Sistema de Análise de Pássaros", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # Frame de upload
        upload_frame = ttk.LabelFrame(main_frame, text="Upload de Imagem", padding="10")
        upload_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Botão de upload
        self.upload_button = ttk.Button(upload_frame, text="Selecionar Imagem", 
                                       command=self.select_image)
        self.upload_button.grid(row=0, column=0, padx=(0, 10))
        
        # Label do arquivo
        self.file_label = ttk.Label(upload_frame, text="Nenhuma imagem selecionada")
        self.file_label.grid(row=0, column=1, sticky=tk.W)
        
        # Frame de análise
        analysis_frame = ttk.LabelFrame(main_frame, text="Análise", padding="10")
        analysis_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        analysis_frame.columnconfigure(0, weight=1)
        analysis_frame.rowconfigure(1, weight=1)
        
        # Botão de análise
        self.analyze_button = ttk.Button(analysis_frame, text="Analisar Imagem", 
                                        command=self.analyze_image, state="disabled")
        self.analyze_button.grid(row=0, column=0, pady=(0, 10))
        
        # Área de resultados
        self.results_text = scrolledtext.ScrolledText(analysis_frame, height=15, width=80)
        self.results_text.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Frame de ações
        actions_frame = ttk.LabelFrame(main_frame, text="Ações", padding="10")
        actions_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        
        # Botão de análise manual
        self.manual_button = ttk.Button(actions_frame, text="Marcar para Análise Manual", 
                                       command=self.mark_for_manual_analysis, state="disabled")
        self.manual_button.grid(row=0, column=0, padx=(0, 10))
        
        # Botão de teste
        self.test_button = ttk.Button(actions_frame, text="Teste de Botão", 
                                     command=self.test_button_function)
        self.test_button.grid(row=0, column=1, padx=(0, 10))
        
        # Botão de limpar
        self.clear_button = ttk.Button(actions_frame, text="Limpar Resultados", 
                                      command=self.clear_results)
        self.clear_button.grid(row=0, column=2)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Pronto")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN)
        status_bar.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
    
    def init_system(self):
        """Inicializar sistema em thread separada"""
        def init_thread():
            try:
                self.status_var.set("Inicializando sistema...")
                self.system = LogicalAIReasoningSystem()
                self.status_var.set("Sistema inicializado com sucesso!")
                self.log_message("✅ Sistema inicializado com sucesso!")
            except Exception as e:
                self.status_var.set(f"Erro ao inicializar: {e}")
                self.log_message(f"❌ Erro ao inicializar sistema: {e}")
        
        threading.Thread(target=init_thread, daemon=True).start()
    
    def select_image(self):
        """Selecionar imagem para análise"""
        filetypes = [
            ("Imagens", "*.jpg *.jpeg *.png *.bmp *.gif"),
            ("JPEG", "*.jpg *.jpeg"),
            ("PNG", "*.png"),
            ("Todos os arquivos", "*.*")
        ]
        
        filename = filedialog.askopenfilename(
            title="Selecionar Imagem",
            filetypes=filetypes
        )
        
        if filename:
            self.current_image_path = filename
            self.file_label.config(text=os.path.basename(filename))
            self.analyze_button.config(state="normal")
            self.log_message(f"📁 Imagem selecionada: {os.path.basename(filename)}")
    
    def analyze_image(self):
        """Analisar imagem"""
        if not self.current_image_path or not self.system:
            messagebox.showerror("Erro", "Selecione uma imagem e aguarde o sistema inicializar!")
            return
        
        def analyze_thread():
            try:
                self.status_var.set("Analisando imagem...")
                self.analyze_button.config(state="disabled")
                
                # Copiar imagem para arquivo temporário
                self.temp_path = f"temp_{os.path.basename(self.current_image_path)}"
                
                # Usar PIL para copiar a imagem
                with Image.open(self.current_image_path) as img:
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img.save(self.temp_path, format='PNG')
                
                self.log_message(f"📸 Imagem temporária criada: {self.temp_path}")
                
                # Analisar com o sistema
                result = self.system.analyze_image_revolutionary(self.temp_path)
                
                # Exibir resultados
                self.display_results(result)
                
                # Habilitar botão de análise manual
                self.manual_button.config(state="normal")
                
                self.status_var.set("Análise concluída!")
                
            except Exception as e:
                self.log_message(f"❌ Erro na análise: {e}")
                self.status_var.set(f"Erro na análise: {e}")
            finally:
                self.analyze_button.config(state="normal")
        
        threading.Thread(target=analyze_thread, daemon=True).start()
    
    def display_results(self, result):
        """Exibir resultados da análise"""
        self.log_message("=" * 50)
        self.log_message("📊 RESULTADOS DA ANÁLISE")
        self.log_message("=" * 50)
        
        # Informações básicas
        self.log_message(f"📁 Arquivo: {result.get('image_path', 'N/A')}")
        self.log_message(f"⏰ Timestamp: {result.get('timestamp', 'N/A')}")
        
        # Análise YOLO
        if 'intuition_analysis' in result and 'yolo_analysis' in result['intuition_analysis']:
            yolo_data = result['intuition_analysis']['yolo_analysis']
            self.log_message(f"🔍 Detecções YOLO: {yolo_data.get('total_detections', 0)}")
            
            for i, detection in enumerate(yolo_data.get('detections', [])):
                self.log_message(f"  {i+1}. {detection.get('class', 'N/A')} - {detection.get('confidence', 0):.2%}")
        
        # Recomendação
        if 'intuition_analysis' in result and 'intuition_analysis' in result['intuition_analysis']:
            intuition_data = result['intuition_analysis']['intuition_analysis']
            self.log_message(f"🧠 Nível de intuição: {intuition_data.get('intuition_level', 'N/A')}")
            self.log_message(f"💡 Recomendação: {intuition_data.get('recommendation', 'N/A')}")
        
        self.log_message("=" * 50)
    
    def mark_for_manual_analysis(self):
        """Marcar imagem para análise manual"""
        if not self.temp_path:
            messagebox.showerror("Erro", "Nenhuma imagem analisada!")
            return
        
        try:
            # LOG DETALHADO DO BOTÃO
            self.button_debug.log_button_click("desktop_manual_analysis", self.temp_path)
            self.button_debug.log_step("INICIANDO PROCESSO DE ANÁLISE MANUAL")
            
            # Verificar se arquivo temporário existe
            file_exists = os.path.exists(self.temp_path)
            self.button_debug.log_file_check(self.temp_path, file_exists)
            
            if not file_exists:
                error_msg = f"Arquivo temporário não encontrado: {self.temp_path}"
                self.log_message(f"❌ {error_msg}")
                self.button_debug.log_error(error_msg, "FILE_NOT_FOUND")
                messagebox.showerror("Erro", error_msg)
            else:
                self.button_debug.log_success("Arquivo temporário encontrado")
                
                # Adicionar à fila de análise manual
                detection_data = {
                    'yolo_detections': [],
                    'confidence': 0.0,
                    'analysis_type': 'desktop_manual',
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
                    self.log_message("✅ Imagem marcada para análise manual!")
                    messagebox.showinfo("Sucesso", "Imagem marcada para análise manual!")
                else:
                    self.button_debug.log_error("Falha ao copiar arquivo!", "COPY_ERROR")
                    self.log_message("❌ Falha ao copiar arquivo!")
                    messagebox.showerror("Erro", "Falha ao copiar arquivo!")
                
                # Remover arquivo temporário após copiar
                if os.path.exists(self.temp_path):
                    os.remove(self.temp_path)
                    self.button_debug.log_file_cleanup(self.temp_path, True)
                    self.log_message("🗑️ Arquivo temporário removido")
                else:
                    self.button_debug.log_file_cleanup(self.temp_path, False)
            
        except Exception as e:
            error_msg = f"Erro ao adicionar para análise manual: {e}"
            self.log_message(f"❌ {error_msg}")
            self.button_debug.log_error(error_msg, "MANUAL_ANALYSIS_ERROR")
            messagebox.showerror("Erro", error_msg)
        finally:
            self.button_debug.log_session_end()
    
    def test_button_function(self):
        """Teste de botão simples"""
        self.log_message("🧪 TESTE DE BOTÃO EXECUTADO!")
        self.log_message("✅ Botão funcionando perfeitamente!")
        messagebox.showinfo("Teste", "Botão funcionando perfeitamente!")
    
    def clear_results(self):
        """Limpar resultados"""
        self.results_text.delete(1.0, tk.END)
        self.log_message("🧹 Resultados limpos")
    
    def log_message(self, message):
        """Adicionar mensagem ao log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.results_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.results_text.see(tk.END)

def main():
    """Função principal"""
    root = tk.Tk()
    app = BirdAnalysisDesktopApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
