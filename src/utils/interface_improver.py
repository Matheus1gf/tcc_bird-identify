#!/usr/bin/env python3
"""
Sistema de Melhorias de Interface - Fase 0
Substitui emojis por ícones Bootstrap profissionais
"""

import re
import os
from typing import Dict, List, Tuple


class InterfaceImprover:
    """Sistema para melhorar a interface removendo emojis e implementando ícones Bootstrap"""
    
    def __init__(self):
        """Inicializar sistema de melhorias de interface"""
        
        # Mapeamento de emojis para ícones Bootstrap
        self.emoji_to_bootstrap = {
            # Ícones principais
            "[BUSCA]": '<i class="bi bi-search"></i>',
            "[ESCRITA]": '<i class="bi bi-sticky"></i>',
            "[ATUALIZACAO]": '<i class="bi bi-arrow-clockwise"></i>',
            "[DADO]": '<i class="bi bi-bar-chart"></i>',
            "[LIMPEZA]": '<i class="bi bi-trash"></i>',
            "[INICIO]": '<i class="bi bi-play"></i>',
            "[PAUSA]": '<i class="bi bi-pause"></i>',
            "[SUCESSO]": '<i class="bi bi-check-circle"></i>',
            "[ERRO]": '<i class="bi bi-x-circle"></i>',
            "[PULAR]": '<i class="bi bi-skip-forward"></i>',
            "[SALVAR]": '<i class="bi bi-save"></i>',
            "[UPLOAD]": '<i class="bi bi-upload"></i>',
            "[INFO]": '<i class="bi bi-info-circle"></i>',
            "[ALERTA]": '<i class="bi bi-exclamation-triangle"></i>',
            "[ALVO]": '<i class="bi bi-bullseye"></i>',
            "[PERFORMANCE]": '<i class="bi bi-graph-up"></i>',
            "[LISTA]": '<i class="bi bi-clipboard"></i>',
            "[CASA]": '<i class="bi bi-house"></i>',
            "[CAMERA]": '<i class="bi bi-camera"></i>',
            "[IA]": '<i class="bi bi-cpu"></i>',
            "[USUARIOS]": '<i class="bi bi-people"></i>',
            "[IDEA]": '<i class="bi bi-lightbulb"></i>',
            "[CONFIG]": '<i class="bi bi-gear"></i>',
            "[IMAGEM]": '<i class="bi bi-image"></i>',
            "[ARQUIVO]": '<i class="bi bi-folder"></i>',
            "[PASSARO]": '<i class="bi bi-sun"></i>',  # Usar sol como pássaro genérico
            "[APRENDIZADO]": '<i class="bi bi-book"></i>',
            ":": '<i class="bi bi-palette"></i>',
            "[MANUTENCAO]": '<i class="bi bi-tools"></i>',
            "[RACIOCINIO]": '<i class="bi bi-chat-text"></i>',
            "[OK]": '<i class="bi bi-check-circle text-success"></i>',
            "[ERRO]": '<i class="bi bi-x-circle text-danger"></i>',
            "[ATENCAO]": '<i class="bi bi-exclamation-triangle text-warning"></i>',
            "[MEDIDA]": '<i class="bi bi-rulers"></i>',
            "[ANALISE]": '<i class="bi bi-microscope"></i>',
            "[TESTE]": '<i class="bi bi-flask"></i>',
            "[GEOMETRICO]": '<i class="bi bi-square"></i>',
            "[RANDOMICO]": '<i class="bi bi-dice-6"></i>',
            "[LINK]": '<i class="bi bi-link"></i>',
            "[WEB]": '<i class="bi bi-globe"></i>',
            "[SEGURANCA]": '<i class="bi bi-lock"></i>',
            "[DOCUMENTO]": '<i class="bi bi-file-text"></i>',
            "[VITORIA]": '<i class="bi bi-trophy"></i>',
            "[QUALIDADE]": '<i class="bi bi-gem"></i>',
            "[ESTRELA]": '<i class="bi bi-star"></i>',
            "[DESTAQUE]": '<i class="bi bi-star-fill"></i>',
            "[QUENTE]": '<i class="bi bi-fire"></i>',
            "[RAPIDO]": '<i class="bi bi-rocket"></i>',
            "[FORTE]": '<i class="bi bi-heart-pulse"></i>',
            "[COLECAO]": '<i class="bi bi-collection"></i>',
            "[NOTIFICACAO]": '<i class="bi bi-bell"></i>',
            "[EMAIL]": '<i class="bi bi-envelope"></i>',
            "[BUSCA]": '<i class="bi bi-search"></i>',
            "[MOBILE]": '<i class="bi bi-phone"></i>',
            "[NOTEBOOK]": '<i class="bi bi-laptop"></i>',
            "[COMPUTADOR]": '<i class="bi bi-display"></i>',
            "[RAPIDO]": '<i class="bi bi-lightning"></i>',
            "[BATERIA]": '<i class="bi bi-battery"></i>',
            "[MUNDIAL]": '<i class="bi bi-globe"></i>',
        }
        
        # Mapeamento para logs (emojis para texto simples)
        self.emoji_to_log_text = {
            "[BUSCA]": "[BUSCA]",
            "[ESCRITA]": "[ESCRITA]",
            "[ATUALIZACAO]": "[ATUALIZACAO]",
            "[DADO]": "[DADO]",
            "[LIMPEZA]": "[LIMPEZA]",
            "[INICIO]": "[INICIO]",
            "[PAUSA]": "[PAUSA]",
            "[SUCESSO]": "[SUCESSO]",
            "[ERRO]": "[ERRO]",
            "[PULAR]": "[PULAR]",
            "[SALVAR]": "[SALVAR]",
            "[UPLOAD]": "[UPLOAD]",
            "[INFO]": "[INFO]",
            "[ALERTA]": "[ALERTA]",
            "[ALVO]": "[ALVO]",
            "[PERFORMANCE]": "[PERFORMANCE]",
            "[LISTA]": "[LISTA]",
            "[CASA]": "[CASA]",
            "[CAMERA]": "[CAMERA]",
            "[IA]": "[IA]",
            "[USUARIOS]": "[USUARIOS]",
            "[IDEA]": "[IDEA]",
            "[CONFIG]": "[CONFIG]",
            "[IMAGEM]": "[IMAGEM]",
            "[ARQUIVO]": "[ARQUIVO]",
            "[PASSARO]": "[PASSARO]",
            "[APRENDIZADO]": "[APRENDIZADO]",
            ":": ":",
            "[MANUTENCAO]": "[MANUTENCAO]",
            "[RACIOCINIO]": "[RACIOCINIO]",
            "[OK]": "[OK]",
            "[ERRO]": "[ERRO]",
            "[ATENCAO]": "[ATENCAO]",
            "[MEDIDA]": "[MEDIDA]",
            "[ANALISE]": "[ANALISE]",
            "[TESTE]": "[TESTE]",
            "[GEOMETRICO]": "[GEOMETRICO]",
            "[RANDOMICO]": "[RANDOMICO]",
            "[LINK]": "[LINK]",
            "[WEB]": "[WEB]",
            "[SEGURANCA]": "[SEGURANCA]",
            "[DOCUMENTO]": "[DOCUMENTO]",
            "[VITORIA]": "[VITORIA]",
            "[QUALIDADE]": "[QUALIDADE]",
            "[ESTRELA]": "[ESTRELA]",
            "[DESTAQUE]": "[DESTAQUE]",
            "[QUENTE]": "[QUENTE]",
            "[RAPIDO]": "[RAPIDO]",
            "[FORTE]": "[FORTE]",
            "[COLECAO]": "[COLECAO]",
            "[NOTIFICACAO]": "[NOTIFICACAO]",
            "[EMAIL]": "[EMAIL]",
            "[MOBILE]": "[MOBILE]",
            "[NOTEBOOK]": "[NOTEBOOK]",
            "[COMPUTADOR]": "[COMPUTADOR]",
            "[RAPIDO]": "[RAPIDO]",
            "[BATERIA]": "[BATERIA]",
            "[MUNDIAL]": "[MUNDIAL]",
        }
    
    def replace_emojis_in_html(self, content: str) -> str:
        """Substitui emojis por ícones Bootstrap em HTML, usando aspas simples para evitar conflitos"""
        emoji_mapping = {}
        
        # Criar mapeamento temporário com aspas simples
        for emoji, bootstrap_icon in self.emoji_to_bootstrap.items():
            # Usar aspas simples dentro do HTML para evitar conflitos
            bootstrap_icon_safe = bootstrap_icon.replace('"', "'")
            emoji_mapping[emoji] = bootstrap_icon_safe
            
        for emoji, bootstrap_icon in emoji_mapping.items():
            content = content.replace(emoji, bootstrap_icon)
        return content
    
    def replace_emojis_in_logs(self, content: str) -> str:
        """Substitui emojis por texto simples em logs"""
        for emoji, log_text in self.emoji_to_log_text.items():
            content = content.replace(emoji, log_text)
        return content
    
    def get_file_type(self, file_path: str) -> str:
        """Determina o tipo do arquivo baseado na extensão"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith('.html'):
            return 'html'
        elif file_path.endswith('.js') or file_path.endswith('.jsx'):
            return 'javascript'
        elif file_path.endswith('.css'):
            return 'css'
        else:
            return 'text'
    
    def process_file(self, file_path: str) -> Tuple[str, int]:
        """Processa um arquivo específico removendo emojis"""
        if not os.path.exists(file_path):
            return "", 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            # Determinar tipo de substituição baseado no arquivo
            file_type = self.get_file_type(file_path)
            
            if file_type == 'python':
                # Para Python, usar texto simples em logs e CSS do Bootstrap em Streamlit
                processed_content = original_content
                
                # Substituir emojis em strings do Streamlit (HTML)
                processed_content = re.sub(
                    r'st\.(title|header|subheader|markdown|write|info|success|warning|error)\s*\(\s*["\'](.*?)["\']\s*\)',
                    lambda m: f'st.{m.group(1)}("{self.replace_emojis_in_html(m.group(2))}")' if any(emoji in m.group(2) for emoji in self.emoji_to_bootstrap.keys()) else m.group(0),
                    processed_content
                )
                
                # Substituir emojis em logs e prints
                for emoji, log_text in self.emoji_to_log_text.items():
                    processed_content = processed_content.replace(emoji, log_text)
                
            else:
                # Para outros arquivos (HTML, CSS, etc.)
                processed_content = self.replace_emojis_in_html(original_content)
            
            # Contar substituições feitas
            original_emoji_count = sum(original_content.count(emoji) for emoji in self.emoji_to_bootstrap.keys())
            processed_emoji_count = sum(processed_content.count(emoji) for emoji in self.emoji_to_bootstrap.keys())
            substitutions = original_emoji_count - processed_emoji_count
            
            # Salvar arquivo processado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            return file_path, substitutions
            
        except Exception as e:
            print(f"Erro ao processar arquivo {file_path}: {e}")
            return "", 0
    
    def process_directory(self, directory_path: str, extensions: List[str] = None) -> Dict[str, int]:
        """Processa todos os arquivos em um diretório"""
        if extensions is None:
            extensions = ['.py', '.html', '.css', '.js', '.jsx']
        
        results = {}
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    filename, substitutions = self.process_file(file_path)
                    if substitutions > 0:
                        results[filename] = substitutions
        
        return results
    
    def add_bootstrap_icons_css(self, file_path: str) -> bool:
        """Adiciona CSS do Bootstrap Icons ao arquivo HTML"""
        if not file_path.endswith('.py'):
            return False
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Verificar se já tem Bootstrap Icons
            if 'bootstrap-icons' in content:
                return False
            
            # CSS para Bootstrap Icons (para Streamlit)
            bootstrap_css = '''
        /* Bootstrap Icons CSS */
        @import url("https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css");
        
        .bi {
            vertical-align: -.125em;
            fill: currentColor;
        }
        '''
            
            # Adicionar CSS antes do estilo atual
            if 'st.markdown("""' in content:
                content = content.replace('st.markdown("""', f'st.markdown("""{bootstrap_css}')
            
            # Salvar arquivo atualizado
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            print(f"Erro ao adicionar Bootstrap Icons ao arquivo {file_path}: {e}")
            return False
    
    def standardize_interface_files(self, directories: List[str]) -> Dict[str, int]:
        """Padroniza arquivos de interface em múltiplos diretórios"""
        total_results = {}
        
        for directory in directories:
            if os.path.exists(directory):
                results = self.process_directory(directory)
                total_results.update(results)
                
                # Adicionar Bootstrap Icons CSS ao web_app.py se existir
                web_app_path = os.path.join(directory, "web_app.py")
                if os.path.exists(web_app_path):
                    self.add_bootstrap_icons_css(web_app_path)
                    print(f"Bootstrap Icons CSS adicionado ao {web_app_path}")
        
        return total_results
    
    def generate_report(self, results: Dict[str, int]) -> str:
        """Gera relatório das melhorias feitas"""
        total_files = len(results)
        total_substitutions = sum(results.values())
        
        report = f"""
=== RELATÓRIO DE MELHORIAS DE INTERFACE ===

Arquivos Processados: {total_files}
Emojis Substituídos: {total_substitutions}

DETALHES POR ARQUIVO:
"""
        
        for file_path, substitutions in sorted(results.items()):
            if substitutions > 0:
                report += f"- {os.path.basename(file_path)}: {substitutions} substituições\n"
        
        report += f"""
ÍCONES BOOTSTRAP IMPLEMENTADOS: {len(self.emoji_to_bootstrap)}

MAPEAMENTO DE EMOJIS PARA ÍCONES:
"""
        
        emoji_count = 0
        for emoji, bootstrap_icon in list(self.emoji_to_bootstrap.items())[:10]:  # Mostrar apenas os primeiros 10
            report += f"{emoji} → {bootstrap_icon}\n"
            emoji_count += 1
        
        if len(self.emoji_to_bootstrap) > 10:
            report += f"... e mais {len(self.emoji_to_bootstrap) - 10} mapeamentos\n"
        
        report += f"""
STATUS: MELHORIAS DE INTERFACE CONCLUÍDAS
DATA: {os.popen('date').read().strip()}
"""
        
        return report


def main():
    """Função principal para aplicar melhorias de interface"""
    print(": Iniciando melhorias de interface - Fase 0")
    
    # Inicializar melhorador
    improver = InterfaceImprover()
    
    # Diretórios a serem processados
    directories_to_process = [
        "src/interfaces",
        "src/core", 
        ".",
        "scripts"
    ]
    
    print("[ARQUIVO] Processando diretórios...")
    
    # Aplicar melhorias
    results = improver.standardize_interface_files(directories_to_process)
    
    # Gerar relatório
    report = improver.generate_report(results)
    
    # Salvar relatório
    with open("interface_improvements_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("[SUCESSO] Melhorias de interface aplicadas com sucesso!")
    print(f"[DADO] {len(results)} arquivos processados")
    print(f"[ATUALIZACAO] {sum(results.values())} emojis substituídos")
    print("📄 Relatório salvo em: interface_improvements_report.txt")


if __name__ == "__main__":
    main()
