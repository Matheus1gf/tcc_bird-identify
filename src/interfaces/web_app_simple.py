import streamlit as st
import os
import sys
import traceback
from PIL import Image
import numpy as np

# Adicionar o diretório raiz ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuração da página - DEVE SER A PRIMEIRA COISA
st.set_page_config(
    page_title="Sistema de Identificação de Pássaros",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Imports dos sistemas
try:
    from src.core.intuition import IntuitionEngine
    from src.utils.debug_logger import DebugLogger
except ImportError as e:
    st.error(f"❌ Erro ao importar sistemas: {e}")
    st.stop()

def main():
    """Função principal simplificada"""
    
    # Título principal
    st.markdown("""
    <h1>🐦 Sistema de Identificação de Pássaros</h1>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # CSS simples
    st.markdown("""
    <style>
        .metric-container {
            text-align: center;
            margin: 10px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Inicializar sistemas de forma ULTRA SIMPLIFICADA
    @st.cache_resource
    def initialize_systems():
        """Inicializa sistemas de forma simplificada para evitar travamentos"""
        try:
            st.info("🔄 Inicializando sistemas básicos...")
            
            # Apenas sistemas essenciais
            debug_logger = DebugLogger()
            intuition_engine = IntuitionEngine("data/models/yolov8n.pt", "data/models/modelo_classificacao_passaros.keras", debug_logger)
            
            st.success("✅ Sistemas básicos inicializados!")
            
            return {
                'debug_logger': debug_logger,
                'intuition_engine': intuition_engine
            }
            
        except Exception as e:
            st.error(f"❌ Erro ao inicializar sistemas: {e}")
            return None
    
    # Inicializar sistemas
    systems = initialize_systems()
    
    if systems is None:
        st.error("❌ Falha na inicialização dos sistemas")
        return
    
    # Extrair sistemas
    debug_logger = systems['debug_logger']
    intuition_engine = systems['intuition_engine']
    
    # Verificar se sistemas essenciais estão disponíveis
    if intuition_engine is None:
        st.error("❌ IntuitionEngine não foi inicializado")
        return
    
    st.success("✅ Sistema inicializado com sucesso!")
    
    # Menu principal com tabs SIMPLIFICADO
    tab_names = [
        '🏠 Início',
        '📸 Análise de Imagem'
    ]
    
    # Criar tabs
    tabs = st.tabs(tab_names)
    inicio_tab, analise_tab = tabs
    
    # TAB 1: INÍCIO - ULTRA SIMPLIFICADO
    with inicio_tab:
        st.markdown("## 🏠 Página Inicial")
        
        st.info("""
        **Sistema de Identificação de Pássaros**
        
        Sistema funcionando com IA para identificar pássaros em imagens.
        
        **Status: ✅ Operacional**
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🧠 Sistema", "✅ Ativo", "100%")
        
        with col2:
            st.metric("🎯 Detecção", "✅ Ativo", "100%")
    
    # TAB 2: ANÁLISE DE IMAGEM - SIMPLIFICADO
    with analise_tab:
        st.markdown("## 📸 Análise de Imagem")
        
        # Upload de imagem - SIMPLIFICADO
        uploaded_file = st.file_uploader(
            "Escolha uma imagem de pássaro",
            type=['jpg', 'jpeg', 'png'],
            help="Faça upload de uma imagem de pássaro para análise"
        )
        
        if uploaded_file is not None:
            try:
                st.info("🔄 Processando imagem...")
                
                # Converter para imagem
                image = Image.open(uploaded_file)
                
                # Exibir imagem
                st.subheader("🖼️ Imagem Carregada")
                st.image(image, width=300)
                
                st.success("✅ Imagem carregada com sucesso!")
                
            except Exception as e:
                st.error(f"❌ Erro ao processar imagem: {str(e)}")
                return
            
            # Botão de análise - ULTRA SIMPLIFICADO
            if st.button("🔍 Analisar Imagem", type="primary", key="analyze_image_btn"):
                with st.spinner("Analisando imagem..."):
                    try:
                        # Salvar imagem temporariamente
                        temp_path = f"temp_{uploaded_file.name}.png"
                        image.save(temp_path)
                        
                        # Análise simples
                        st.info("🔄 Iniciando análise...")
                        results = intuition_engine.analyze_image_intuition(temp_path)
                        
                        # Exibir resultados simples
                        st.markdown("### 📊 Resultados da Análise")
                        
                        if results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("🎯 Pássaro Detectado", "✅ Sim" if results.get('detected', False) else "❌ Não")
                                st.metric("📊 Confiança", f"{results.get('confidence', 0):.2f}")
                            
                            with col2:
                                st.metric("🏷️ Espécie", results.get('species', 'Desconhecida'))
                                st.metric("⚡ Tempo", f"{results.get('processing_time', 0):.2f}s")
                            
                            # Detalhes adicionais
                            if 'details' in results:
                                st.markdown("### 📋 Detalhes")
                                st.json(results['details'])
                        
                        else:
                            st.warning("⚠️ Nenhum resultado retornado")
                        
                        # Limpar arquivo temporário
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante análise: {str(e)}")
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
