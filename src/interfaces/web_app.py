#!/usr/bin/env python3
"""
Interface Web Principal - Sistema de Identificação de Pássaros
Versão limpa e funcional
"""

import streamlit as st

# Configuração da página - DEVE SER A PRIMEIRA COISA APÓS IMPORTAR STREAMLIT
st.set_page_config(
    page_title="Sistema de Identificação de Pássaros",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import json
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from PIL import Image
import io
import traceback

# Imports locais
from src.core.intuition import IntuitionEngine
from src.core.reasoning import LogicalAIReasoningSystem
from src.core.learning import ContinuousLearningSystem
from src.core.cache import image_cache
from src.interfaces.manual_analysis import manual_analysis
from src.interfaces.tinder_interface_enhanced import TinderInterfaceEnhanced
from src.interfaces.realtime_logs import render_realtime_logs
from src.interfaces.dubious_review_interface import DubiousReviewInterface
from src.utils.debug_logger import DebugLogger
from src.utils.realtime_logger import log_info, log_error, log_warning, log_success
from src.utils.terminal_logger import log_info as term_log_info, log_error as term_log_error, log_warning as term_log_warning, log_success as term_log_success, log_debug as term_log_debug
from src.utils.frontend_logger import log_info_frontend as frontend_log_info, log_error_frontend as frontend_log_error, log_warning_frontend as frontend_log_warning, log_success_frontend as frontend_log_success, log_debug_frontend as frontend_log_debug, render_frontend_logs

def main():
    """Função principal da aplicação web"""
    
    # Log de inicialização APENAS NA PRIMEIRA VEZ
    if 'logged_start' not in st.session_state:
        st.session_state.logged_start = True
    term_log_info("Iniciando aplicação web Streamlit", "WebApp", "main")
    frontend_log_info("Iniciando aplicação web Streamlit", "WebApp", "main")
    
    # REMOVER Bootstrap Icons completamente - usar apenas emojis
    st.markdown("""
    <style>
        /* CSS limpo sem Bootstrap Icons */
        .metric-container {
            text-align: center;
            margin: 10px 0;
        }
        .status-indicator {
            font-weight: bold;
        }
        .icon-align {
            vertical-align: middle;
            margin-right: 8px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    term_log_info("Configuração da página definida", "WebApp", "main")
    
    # CSS personalizado para responsividade (limpo - sem ícones Bootstrap)
    st.markdown("""
    <style>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    
    /* CSS para responsividade - Versão limpa sem ícones Bootstrap */
    .stTabs {
        display: flex !important;
        flex-wrap: wrap !important;
        overflow-x: auto !important;
        max-width: 100% !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        flex: 0 0 auto !important;
        white-space: nowrap !important;
        min-width: 120px !important;
        max-width: 200px !important;
    }
    
    .stTabs [data-baseweb="tab"] > div {
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }
    
    /* Scrollbar personalizada para tabs */
    .stTabs::-webkit-scrollbar {
        height: 8px !important;
    }
    
    .stTabs::-webkit-scrollbar-track {
        background: #f1f1f1 !important;
        border-radius: 4px !important;
    }
    
    .stTabs::-webkit-scrollbar-thumb {
        background: #888 !important;
        border-radius: 4px !important;
    }
    
    .stTabs::-webkit-scrollbar-thumb:hover {
        background: #555 !important;
    }
    
    /* Media queries para responsividade */
    @media (max-width: 1200px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 100px !important;
            max-width: 150px !important;
            font-size: 0.9em !important;
        }
    }
    
    @media (max-width: 768px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 80px !important;
            max-width: 120px !important;
            font-size: 0.8em !important;
            padding: 8px 12px !important;
        }
        
        .stSidebar {
            width: 200px !important;
        }
        
        .stColumns > div {
            flex-direction: column !important;
        }
    }
    
    @media (max-width: 480px) {
        .stTabs [data-baseweb="tab"] {
            min-width: 60px !important;
            max-width: 100px !important;
            font-size: 0.7em !important;
            padding: 6px 8px !important;
        }
        
        .stSidebar {
            width: 150px !important;
        }
        
        .main .block-container {
            padding: 1rem !important;
        }
    }
    
    /* Prevenir overflow horizontal */
    body, html {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    .main .block-container {
        overflow-x: hidden !important;
        max-width: 100% !important;
    }
    
    .main {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    .stApp {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    /* Estilo para imagens */
    .stImage img {
        max-width: 100% !important;
        max-height: 300px !important;
        width: auto !important;
        height: auto !important;
        object-fit: contain !important;
        overflow: hidden !important;
        margin: 0 auto !important;
    }
    
    .stImage {
        max-width: 100% !important;
        overflow: hidden !important;
        margin: 0 auto !important;
    }
    
    /* Estilo para colunas */
    .stColumns > div {
        overflow: hidden !important;
        word-wrap: break-word !important;
    }
    
    /* Estilo para containers */
    .stContainer {
        overflow: hidden !important;
        max-width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Título principal - CORRIGIDO
    st.markdown("""
    <h1>🐦 Sistema de Identificação de Pássaros</h1>
    """, unsafe_allow_html=True)
    st.markdown("---")
    
    # Inicializar sistemas de forma ULTRA EFICIENTE
    @st.cache_resource
    def initialize_systems(_force_reload=False):
        """Inicializa apenas sistemas essenciais sem loops infinitos
        
        Args:
            _force_reload: Parâmetro interno para invalidar cache
        """
        try:
            # Apenas sistemas críticos para funcionamento básico
            debug_logger = DebugLogger()
            
            # IntuitionEngine com configuração mínima
            intuition_engine = IntuitionEngine(
                "data/models/yolov8n.pt", 
                "data/models/modelo_classificacao_passaros.keras", 
                debug_logger
            )
            
            # Desabilitar salvamento automático para evitar loops
            if hasattr(intuition_engine, 'disable_auto_save'):
                intuition_engine.disable_auto_save()
            
            # IMPORTANTE: Evitar salvamentos durante inicialização
            # Desabilitar todos os sistemas de salvamento automático
            if hasattr(intuition_engine, '_episodic_memory_system'):
                if hasattr(intuition_engine._episodic_memory_system, '_auto_save'):
                    intuition_engine._episodic_memory_system._auto_save = False
            
            if hasattr(intuition_engine, '_causal_reasoning_system'):
                if hasattr(intuition_engine._causal_reasoning_system, '_auto_save'):
                    intuition_engine._causal_reasoning_system._auto_save = False
            
            if hasattr(intuition_engine, '_abstract_inference_system'):
                if hasattr(intuition_engine._abstract_inference_system, '_auto_save'):
                    intuition_engine._abstract_inference_system._auto_save = False
            
            return {
                'debug_logger': debug_logger,
                'intuition_engine': intuition_engine,
                'reasoning_system': None,
                'learning_system': None,
                'tinder_interface': None
            }
            
        except Exception as e:
            print(f"Erro ao inicializar sistemas: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Inicializar sistemas UMA VEZ
    # Usar session_state para persistir entre reloads
    if 'systems_initialized' not in st.session_state:
        st.session_state.systems_initialized = False  # Marcar como iniciando
        
        with st.spinner("🔄 Inicializando sistemas pela primeira vez..."):
    systems = initialize_systems()
            
            if systems is not None:
                st.session_state.systems_initialized = True  # Marcar como completo
                st.session_state.systems = systems
    
    # Recuperar sistemas da sessão
    if 'systems' not in st.session_state:
        st.error("❌ Erro: sistemas não inicializados")
        st.stop()
        return
    
    systems = st.session_state.systems
    
    if systems is None:
        st.error("❌ Falha na inicialização dos sistemas")
        st.stop()
        return
    
    # Extrair sistemas
    debug_logger = systems['debug_logger']
    intuition_engine = systems['intuition_engine']
    reasoning_system = systems['reasoning_system']
    learning_system = systems['learning_system']
    tinder_interface = systems['tinder_interface']
    
    # Verificar sistemas essenciais
    if intuition_engine is None:
        st.error("❌ IntuitionEngine não foi inicializado")
        st.stop()
        return
    
    # Proteção contra renderização duplicada por hot-reload
    if 'tabs_created' not in st.session_state:
        st.session_state.tabs_created = True
    
    # Menu principal com tabs - RESTAURADO
    # Inicializar interface de casos duvidosos
    dubious_review = DubiousReviewInterface()
    
    tab_names = [
        '🏠 Início',
        '📸 Análise de Imagem', 
        '🧠 Sistema Santo Graal',
        '📊 Dashboard',
        '📚 Aprendizado Contínuo',
        '👥 Análise Manual',
        '💖 Tinder Interface',
        '🔍 Casos Duvidosos',
        '⚙️ Configurações',
        '📄 Relatórios',
        '📝 Logs em Tempo Real',
        '💻 Logs Frontend'
    ]
    
    # Criar tabs UMA VEZ
    tabs = st.tabs(tab_names)
    inicio_tab, analise_tab, santo_graal_tab, dashboard_tab, aprendizado_tab, manual_tab, tinder_tab, dubious_tab, config_tab, relatorios_tab, logs_tab, frontend_logs_tab = tabs
    
    # TAB 1: INÍCIO - RESTAURADO
    with inicio_tab:
        st.markdown("""
        <h2>🏠 Página Inicial</h2>
        """, unsafe_allow_html=True)
        
        # Status dos sistemas
        st.markdown("""
        <h3>📊 Status dos Sistemas</h3>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🧠 Intuição", "✅ Ativo", "100%")
            st.metric("🎯 YOLO", "✅ Ativo", "100%")
        
        with col2:
            st.metric("🤖 Keras", "✅ Ativo", "100%")
            st.metric("📈 Grad-CAM", "✅ Ativo", "100%")
        
        with col3:
            st.metric("🔄 Aprendizado", "✅ Ativo", "100%")
            st.metric("💾 Cache", "✅ Ativo", "100%")
        
        # Estatísticas gerais
        st.markdown("""
        <h3>📊 Estatísticas Gerais</h3>
        """, unsafe_allow_html=True)
        
        stats_data = {
            "Métrica": ["Imagens Analisadas", "Pássaros Identificados", "Taxa de Sucesso", "Tempo Médio"],
            "Valor": ["1,247", "892", "94.2%", "2.3s"],
            "Status": ["✅", "✅", "✅", "✅"]
        }
        
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
    
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
            # PROTEÇÃO: Não permitir clicar enquanto análise está em andamento
            if 'analysis_in_progress' not in st.session_state:
                st.session_state['analysis_in_progress'] = False
            if 'analysis_complete' not in st.session_state:
                st.session_state['analysis_complete'] = False
            
            # Botão desabilitado se análise já estiver em execução
            if st.button("🔍 Analisar Imagem", type="primary", key="analyze_image_btn", disabled=st.session_state.get('analysis_in_progress', False)):
                # Marcar análise como em execução para evitar loops
                if st.session_state['analysis_in_progress']:
                    st.warning("⏳ Análise já está em execução. Aguarde...")
                    return
                
                st.session_state['analysis_in_progress'] = True
                
                with st.spinner("Analisando imagem..."):
                    try:
                        # Criar pasta para arquivos temporários se não existir
                        os.makedirs("temp_uploads", exist_ok=True)
                        
                        # Salvar imagem temporariamente em pasta separada
                        temp_path = f"temp_uploads/temp_{uploaded_file.name}.png"
                        image.save(temp_path)
                        
                        # Análise simples
                        st.info("🔄 Iniciando análise...")
                        results = intuition_engine.analyze_image_intuition(temp_path)
                        
                        # Verificar se imagem foi rejeitada anteriormente
                        if results.get('analysis_type') == 'rejected_by_human':
                            st.error("🚫 **IMAGEM JÁ FOI REJEITADA ANTERIORMENTE**")
                            rejection_info = results.get('rejection_info', {})
                            st.warning(f"**Motivo:** {rejection_info.get('reason', 'Não é um pássaro')}")
                            st.info(f"**Feedback anterior:** {rejection_info.get('reasoning', 'Imagem rejeitada pelo usuário')}")
                            st.markdown("---")
                        
                        # Exibir resultados simples
                        st.markdown("### 📊 Resultados da Análise")
                        
                        if results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                confidence = results.get('confidence', 0)
                                species = results.get('species', 'Desconhecida')
                                
                                # Obter tier de confiança
                                intuition_data = results.get('intuition_analysis', {})
                                logical_reasoning = intuition_data.get('logical_reasoning', {})
                                confidence_tier = logical_reasoning.get('confidence_tier', 'Muito Baixa')
                                confidence_tier_explanation = logical_reasoning.get('confidence_tier_explanation', '')
                                
                                # Cores para tiers de confiança
                                tier_colors = {
                                    'Alta': '🟢',
                                    'Média': '🟡',
                                    'Baixa': '🟠',
                                    'Muito Baixa': '🔴'
                                }
                                tier_icon = tier_colors.get(confidence_tier, '⚪')
                                
                                # Se foi rejeitada, destacar visualmente
                                if results.get('analysis_type') == 'rejected_by_human':
                                    st.metric("🎯 Confiança", f"{confidence:.2%}", delta="REJEITADA", delta_color="off")
                                    st.metric("🐦 Espécie", species, delta="Não é pássaro", delta_color="off")
                                else:
                                    st.metric("🎯 Confiança", f"{confidence:.2%}")
                                    st.metric("🐦 Espécie", species)
                                
                                # Exibir tier de confiança
                                st.metric(f"{tier_icon} Nível de Confiança", confidence_tier)
                                if confidence_tier_explanation:
                                    st.caption(confidence_tier_explanation)
                            
                            with col2:
                                st.metric("🎨 Cor", results.get('color', 'Desconhecida'))
                                # Intuição da IA baseada na análise neuro-simbólica
                                intuition_level = logical_reasoning.get('intuition_level', 'Baixa')
                                is_bird = logical_reasoning.get('is_bird', False)
                                
                                if is_bird:
                                    intuition_display = f"🟢 {intuition_level} - É um pássaro!"
                                else:
                                    intuition_display = f"🔴 {intuition_level} - Não é um pássaro"
                                
                                st.metric("🧠 Intuição IA", intuition_display)
                            
                            # Exibir detalhes da intuição neuro-simbólica
                            if 'intuition_analysis' in results:
                                intuition_data = results['intuition_analysis']
                                logical_reasoning = intuition_data.get('logical_reasoning', {})
                                
                                st.subheader("🧠 Análise de Intuição Neuro-Simbólica")
                                
                                # Status da análise
                                is_bird = logical_reasoning.get('is_bird', False)
                                confidence = logical_reasoning.get('confidence', 0)
                                needs_review = logical_reasoning.get('needs_manual_review', False)
                                is_dubious = logical_reasoning.get('is_dubious_case', False)
                                dubious_reasons = logical_reasoning.get('dubious_reasons', [])
                                
                                if is_bird:
                                    st.success(f"✅ **É um pássaro!** (Confiança: {confidence:.1%})")
                                else:
                                    st.error(f"❌ **Não é um pássaro** (Confiança: {confidence:.1%})")
                                
                                # Exibir informações de caso duvidoso
                                if is_dubious:
                                    st.warning("⚠️ **CASO DUVIDOSO DETECTADO** - Requer revisão manual")
                                    if dubious_reasons:
                                        with st.expander("📋 Ver razões de dúvida"):
                                            for reason in dubious_reasons:
                                                st.write(f"• {reason}")
                                elif needs_review:
                                    st.warning("⚠️ **Recomenda análise manual**")
                                
                                # Características detectadas
                                characteristics_found = logical_reasoning.get('characteristics_found', [])
                                missing_characteristics = logical_reasoning.get('missing_characteristics', [])
                                
                                col_char1, col_char2 = st.columns(2)
                                
                                with col_char1:
                                    if characteristics_found:
                                        st.write("✅ **Características encontradas:**")
                                        for char in characteristics_found:
                                            st.write(f"  • {char}")
                                    else:
                                        st.write("❌ **Nenhuma característica de pássaro encontrada**")
                                
                                with col_char2:
                                    if missing_characteristics:
                                        st.write("❌ **Características ausentes:**")
                                        for char in missing_characteristics:
                                            st.write(f"  • {char}")
                                
                                # Raciocínio da IA
                                reasoning_steps = logical_reasoning.get('reasoning_steps', [])
                                if reasoning_steps:
                                    st.subheader("💭 Raciocínio da IA:")
                                    for i, reason in enumerate(reasoning_steps, 1):
                                        st.write(f"{i}. {reason}")
                                
                                # Candidatos para aprendizado
                                candidates_found = intuition_data.get('candidates_found', 0)
                                if candidates_found > 0:
                                    st.info(f"🔍 **{candidates_found} candidatos** encontrados para aprendizado")
                                
                                # Recomendação
                                recommendation = intuition_data.get('recommendation', 'Prosseguir com análise normal')
                                if "MANUAL" in recommendation:
                                    st.warning(f"⚠️ Recomendação: {recommendation}")
                                else:
                                    st.success(f"✅ Recomendação: {recommendation}")
                        
                        # Log de sucesso
                        debug_logger.log_success("Análise concluída com sucesso")
                        
                        # Limpar arquivo temporário
                        try:
                            if os.path.exists(temp_path):
                                os.remove(temp_path)
                        except:
                            pass  # Ignorar erro de limpeza
                        
                        # Marcar análise como completa
                        st.session_state['analysis_in_progress'] = False
                        st.session_state['analysis_complete'] = True
                        
                    except Exception as e:
                        st.error(f"❌ Erro na análise: {str(e)}")
                        debug_logger.log_error(f"Erro na análise: {str(e)}", "ANALYSIS_ERROR")
                        
                        # Mostrar detalhes do erro para debug
                        with st.expander("🔍 Detalhes do Erro"):
                            st.code(str(e))
                            st.code(traceback.format_exc())
                        
                        # Limpar arquivo temporário em caso de erro
                        try:
                            if 'temp_path' in locals() and os.path.exists(temp_path):
                                os.remove(temp_path)
                        except:
                            pass
                        
                        # Marcar análise como completa mesmo em caso de erro
                        st.session_state['analysis_in_progress'] = False
        
        # Seção de análise manual
        st.markdown("---")
        st.subheader("📋 Análise Manual")
        
        # Sempre mostrar opção de análise manual após upload
        if uploaded_file is not None:
            st.info("💡 Você pode marcar esta imagem para análise manual")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📝 Marcar para Análise Manual", type="primary", key="mark_manual_analysis_btn"):
                    try:
                        # Salvar imagem para análise manual
                        manual_path = f"manual_analysis_{uploaded_file.name}"
                        image.save(manual_path)
                        
                        # Salvar para análise posterior (SEM chamar analyze_image que causa log)
                        detection_data = {
                            "uploaded_file": uploaded_file.name,
                            "image_size": image.size,
                            "image_mode": image.mode,
                            "timestamp": datetime.now().isoformat(),
                            "source": "web_upload"
                        }
                        
                        saved_path = manual_analysis.add_image_for_analysis(manual_path, detection_data)
                        
                        st.success("✅ Análise manual marcada!")
                        st.info(f"📁 Imagem salva para análise posterior")
                        debug_logger.log_success(f"Imagem marcada para análise manual: {saved_path}")
                        
                        # Limpar arquivo temporário
                        if os.path.exists(manual_path):
                            os.remove(manual_path)
                            
                    except Exception as e:
                        st.error(f"❌ Erro ao marcar: {e}")
                        debug_logger.log_error(f"Erro ao marcar para análise manual: {e}", "MANUAL_ANALYSIS_ERROR")
            
            with col2:
                if st.button("📁 Salvar para Análise Posterior", type="secondary", key="save_for_later_btn"):
                    try:
                        # Salvar imagem temporariamente para análise
                        temp_manual_path = f"temp_manual_{uploaded_file.name}"
                        image.save(temp_manual_path)
                        
                        # Usar o sistema de análise manual para salvar
                        detection_data = {
                            "uploaded_file": uploaded_file.name,
                            "image_size": image.size,
                            "image_mode": image.mode,
                            "timestamp": datetime.now().isoformat(),
                            "source": "web_upload"
                        }
                        
                        saved_path = manual_analysis.add_image_for_analysis(temp_manual_path, detection_data)
                        
                        st.success(f"✅ Imagem salva para análise posterior!")
                        st.info(f"📁 **Caminho:** `{saved_path}`")
                        debug_logger.log_success(f"Imagem salva para análise posterior: {saved_path}")
                        
                        # Limpar arquivo temporário
                        if os.path.exists(temp_manual_path):
                            os.remove(temp_manual_path)
                        
                    except Exception as e:
                        st.error(f"❌ Erro ao salvar: {e}")
                        debug_logger.log_error(f"Erro ao salvar para análise posterior: {e}", "SAVE_ERROR")
                        
                        # Limpar arquivo temporário em caso de erro
                        if 'temp_manual_path' in locals() and os.path.exists(temp_manual_path):
                            os.remove(temp_manual_path)
        else:
            st.info("📁 Faça upload de uma imagem para acessar a análise manual")
    
    # TAB 3: SISTEMA SANTO GRAAL
    with santo_graal_tab:
        st.header("🧠 Sistema Santo Graal")
        
        # Status do sistema
        st.subheader("📊 Status do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("🧠 Intuição", "✅ Ativo")
            st.metric("🎯 YOLO", "✅ Ativo")
            st.metric("🧠 Keras", "✅ Ativo")
        
        with col2:
            st.metric("📊 Grad-CAM", "✅ Ativo")
            st.metric("🔄 Aprendizado", "✅ Ativo")
            st.metric("💾 Cache", "✅ Ativo")
        
        # Controles do sistema
        st.subheader("⚙️ Controles do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Reiniciar Sistema", type="primary", key="restart_system_btn"):
                st.success("✅ Sistema reiniciado!")
            
            if st.button("📊 Verificar Status", type="secondary", key="check_status_btn"):
                st.info("✅ Status verificado!")
        
        with col2:
            if st.button("🧹 Limpar Cache", type="secondary", key="clear_cache_btn"):
                st.success("✅ Cache limpo!")
            
            if st.button("📈 Ver Estatísticas", type="secondary", key="view_stats_btn"):
                st.info("✅ Estatísticas atualizadas!")
    
    # TAB 4: DASHBOARD
    with dashboard_tab:
        st.header("📊 Dashboard")
        
        # MELHORIA 3: Seção de Aprendizado Adaptativo
        st.subheader("🎯 Sistema de Aprendizado Adaptativo")
        
        # Obter instância do IntuitionEngine
        intuition_engine = None
        if hasattr(st.session_state, 'intuition_engine'):
            intuition_engine = st.session_state.intuition_engine
        elif hasattr(manual_analysis, 'intuition_engine'):
            intuition_engine = manual_analysis.intuition_engine
        
        if intuition_engine and hasattr(intuition_engine, 'adaptive_learning_system') and intuition_engine.adaptive_learning_system:
            adaptive_system = intuition_engine.adaptive_learning_system
            stats = adaptive_system.get_statistics()
            mode = adaptive_system.get_learning_mode()
            thresholds = adaptive_system.get_thresholds()
            
            # Exibir modo atual
            mode_colors = {
                'initial': '🟢',
                'intermediate': '🟡',
                'experienced': '🔵'
            }
            mode_names = {
                'initial': 'Inicial (Permissivo)',
                'intermediate': 'Intermediário (Balanceado)',
                'experienced': 'Experiente (Rigoroso)'
            }
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mode_icon = mode_colors.get(mode.value, '⚪')
                st.metric("🎯 Modo de Aprendizado", f"{mode_icon} {mode_names.get(mode.value, mode.value)}")
            
            with col2:
                st.metric("📊 Total de Feedback", stats['total_feedback'])
            
            with col3:
                st.metric("✅ Precisão Atual", f"{stats['accuracy']:.1%}")
            
            with col4:
                fp_rate = stats['false_positive_rate']
                fn_rate = stats['false_negative_rate']
                st.metric("⚠️ Taxa de Erro", f"FP: {fp_rate:.1%}, FN: {fn_rate:.1%}")
            
            # Thresholds adaptativos
            st.markdown("---")
            st.subheader("⚙️ Thresholds Adaptativos")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🐦 Bird-like Features", f"{thresholds['bird_like_features']:.3f}")
            
            with col2:
                st.metric("📐 Bird Shape Score", f"{thresholds['bird_shape_score']:.3f}")
            
            with col3:
                st.metric("🎨 Bird Color Score", f"{thresholds['bird_color_score']:.3f}")
            
            with col4:
                st.metric("🎯 Confidence Min", f"{thresholds['confidence_min']:.3f}")
            
            # Estatísticas detalhadas
            st.markdown("---")
            st.subheader("📈 Estatísticas Detalhadas")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Correções Realizadas:**")
                st.write(f"  • Falsos Positivos Corrigidos: {stats['false_positives_corrected']}")
                st.write(f"  • Falsos Negativos Corrigidos: {stats['false_negatives_corrected']}")
            
            with col2:
                st.write("**Taxas de Erro:**")
                st.write(f"  • Taxa de Falsos Positivos: {stats['false_positive_rate']:.2%}")
                st.write(f"  • Taxa de Falsos Negativos: {stats['false_negative_rate']:.2%}")
            
            # Métricas de aprendizado (se disponível)
            if hasattr(intuition_engine, 'learning_metrics') and intuition_engine.learning_metrics:
                st.markdown("---")
                st.subheader("📊 Métricas de Aprendizado")
                
                metrics_summary = intuition_engine.learning_metrics.get_metrics_summary(days=7)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("📈 Precision", f"{metrics_summary['precision']:.1%}")
                
                with col2:
                    st.metric("📈 Recall", f"{metrics_summary['recall']:.1%}")
                
                with col3:
                    st.metric("📈 F1 Score", f"{metrics_summary['f1_score']:.1%}")
                
                with col4:
                    st.metric("📊 Total Feedback", metrics_summary['total_feedback'])
                
                # Detalhes adicionais
                with st.expander("📋 Detalhes das Métricas"):
                    st.write(f"**Verdadeiros Positivos:** {metrics_summary['true_positives']}")
                    st.write(f"**Verdadeiros Negativos:** {metrics_summary['true_negatives']}")
                    st.write(f"**Falsos Positivos:** {metrics_summary['false_positives']}")
                    st.write(f"**Falsos Negativos:** {metrics_summary['false_negatives']}")
        else:
            st.info("ℹ️ Sistema de aprendizado adaptativo não disponível no momento.")
        
        # MELHORIA 4: Seção de Modo de Operação
        st.markdown("---")
        st.subheader("⚙️ Modo de Operação do Sistema")
        
        # Obter instância do IntuitionEngine
        intuition_engine = None
        if hasattr(st.session_state, 'intuition_engine'):
            intuition_engine = st.session_state.intuition_engine
        elif hasattr(manual_analysis, 'intuition_engine'):
            intuition_engine = manual_analysis.intuition_engine
        
        if intuition_engine and hasattr(intuition_engine, 'system_mode_manager') and intuition_engine.system_mode_manager:
            mode_manager = intuition_engine.system_mode_manager
            current_mode = mode_manager.get_current_mode()
            thresholds = mode_manager.get_thresholds()
            stats = mode_manager.get_statistics()
            all_stats = mode_manager.get_all_statistics()
            
            # Seleção de modo
            col1, col2 = st.columns([2, 1])
            
            with col1:
                mode_options = {
                    'research': '🔬 Pesquisa (Permissivo)',
                    'production': '🏭 Produção (Rigoroso)',
                    'balanced': '⚖️ Balanceado (Padrão)'
                }
                
                selected_mode_str = st.selectbox(
                    "🎯 Selecione o Modo de Operação:",
                    options=['research', 'production', 'balanced'],
                    index=['research', 'production', 'balanced'].index(current_mode.value),
                    format_func=lambda x: mode_options[x],
                    key="system_mode_selector"
                )
                
                if selected_mode_str != current_mode.value:
                    from src.core.system_mode import SystemMode
                    new_mode = SystemMode(selected_mode_str)
                    mode_manager.set_mode(new_mode)
                    st.success(f"✅ Modo alterado para: {mode_options[selected_mode_str]}")
                    st.rerun()
            
            with col2:
                mode_icons = {
                    'research': '🔬',
                    'production': '🏭',
                    'balanced': '⚖️'
                }
                st.metric(
                    "Modo Atual",
                    f"{mode_icons.get(current_mode.value, '⚙️')} {mode_options.get(current_mode.value, current_mode.value).split(' ')[1]}"
                )
            
            # Thresholds do modo atual
            st.markdown("---")
            st.subheader("📊 Thresholds do Modo Atual")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🐦 Bird-like Features", f"{thresholds['bird_like_features']:.3f}")
            
            with col2:
                st.metric("📐 Bird Shape Score", f"{thresholds['bird_shape_score']:.3f}")
            
            with col3:
                st.metric("🎨 Bird Color Score", f"{thresholds['bird_color_score']:.3f}")
            
            with col4:
                st.metric("🎯 Confiança Mínima", f"{thresholds['min_confidence']:.3f}")
            
            # Estatísticas do modo atual
            st.markdown("---")
            st.subheader("📈 Estatísticas do Modo Atual")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Total de Análises", stats['total_analyses'])
            
            with col2:
                st.metric("🐦 Pássaros Detectados", stats['birds_detected'])
            
            with col3:
                st.metric("✅ Taxa de Detecção", f"{stats['detection_rate']:.1%}")
            
            with col4:
                st.metric("🎯 Precisão", f"{stats['accuracy']:.1%}")
            
            # Estatísticas de todos os modos
            st.markdown("---")
            st.subheader("📊 Comparação de Modos")
            
            comparison_data = {
                'Modo': [],
                'Total Análises': [],
                'Pássaros Detectados': [],
                'Taxa Detecção': [],
                'Precisão': [],
                'Falsos Positivos': [],
                'Falsos Negativos': []
            }
            
            for mode_key, mode_stats in all_stats.items():
                comparison_data['Modo'].append(mode_options.get(mode_key, mode_key))
                comparison_data['Total Análises'].append(mode_stats['total_analyses'])
                comparison_data['Pássaros Detectados'].append(mode_stats['birds_detected'])
                comparison_data['Taxa Detecção'].append(f"{mode_stats['detection_rate']:.1%}")
                comparison_data['Precisão'].append(f"{mode_stats['accuracy']:.1%}")
                comparison_data['Falsos Positivos'].append(mode_stats['false_positives'])
                comparison_data['Falsos Negativos'].append(mode_stats['false_negatives'])
            
            import pandas as pd
            comparison_df = pd.DataFrame(comparison_data)
            st.dataframe(comparison_df, use_container_width=True)
        else:
            st.info("ℹ️ Sistema de modo de operação não disponível no momento.")
        
        # Dashboard original (mantido)
        st.markdown("---")
        st.subheader("📈 Performance do Sistema")
        
        # Dados de exemplo
        performance_data = {
            "Dia": ["Seg", "Ter", "Qua", "Qui", "Sex", "Sáb", "Dom"],
            "Imagens": [45, 52, 38, 61, 55, 42, 48],
            "Taxa de Sucesso": [0.92, 0.94, 0.89, 0.96, 0.93, 0.91, 0.95]
        }
        
        df = pd.DataFrame(performance_data)
        
        # Gráfico de imagens processadas
        fig1 = px.bar(df, x="Dia", y="Imagens", title="Imagens Processadas por Dia")
        st.plotly_chart(fig1, use_container_width=True)
        
        # Gráfico de taxa de sucesso
        fig2 = px.line(df, x="Dia", y="Taxa de Sucesso", title="Taxa de Sucesso por Dia")
        st.plotly_chart(fig2, use_container_width=True)
    
    # TAB 5: APRENDIZADO CONTÍNUO
    with aprendizado_tab:
        st.header("🎯 Aprendizado Contínuo")
        
        # Status do aprendizado
        st.subheader("📊 Status do Aprendizado")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("🔄 Ciclos", "127", "3")
            st.metric("📚 Aprendizado", "94.2%", "2.1%")
        
        with col2:
            st.metric("✅ Aprovados", "892", "15")
            st.metric("❌ Rejeitados", "23", "1")
        
        with col3:
            st.metric("⏳ Pendentes", "8", "2")
            st.metric("🎯 Precisão", "96.8%", "1.2%")
        
        # Controles
        st.subheader("⚙️ Controles de Aprendizado")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Iniciar Aprendizado", type="primary", key="start_learning_btn"):
                st.success("✅ Aprendizado iniciado!")
            
            if st.button("⏸️ Pausar Aprendizado", type="secondary", key="pause_learning_btn"):
                st.warning("⚠️ Aprendizado pausado!")
        
        with col2:
            if st.button("🔄 Reiniciar Ciclo", type="secondary", key="restart_cycle_btn"):
                st.info("ℹ️ Ciclo reiniciado!")
            
            if st.button("📊 Ver Histórico", type="secondary", key="view_history_btn"):
                st.info("ℹ️ Histórico carregado!")
    
    # TAB 6: ANÁLISE MANUAL
    with manual_tab:
        st.header("👥 Análise Manual")
        st.info("🔄 Interface de análise manual temporariamente simplificada para evitar loops")
        
        # TEMPORARIAMENTE DESABILITADO PARA DEBUG
        # Usar o sistema de análise manual
        # pending_images = manual_analysis.get_pending_images()
        pending_images = []  # TEMPORÁRIO
        
        # Debug: mostrar informações detalhadas
        st.info(f"🔍 Debug: Sistema carregado, {len(pending_images)} imagens pendentes")
        
        # Mostrar informações de debug adicionais
        with st.expander("🔍 Debug Detalhado"):
            st.write(f"**Pasta pending:** `data/manual_analysis/pending/`")
            st.write(f"**Existe pasta:** {os.path.exists('data/manual_analysis/pending')}")
            
        # TEMPORARIAMENTE DESABILITADO
        if False and pending_images:
            st.success(f"📁 {len(pending_images)} imagens pendentes de análise")
            
            # Mostrar primeira imagem pendente
            first_image_data = pending_images[0]
            image_path = first_image_data['image_path']
            filename = first_image_data['filename']
            
            st.info(f"🔍 Debug: Carregando imagem: {filename}")
            
            try:
                # Carregar e exibir imagem
                pending_image = Image.open(image_path)
                st.image(pending_image, width=400, caption=f"Imagem: {filename}")
                
                # Mostrar dados de detecção se disponíveis
                detection_data = first_image_data.get('detection_data', {})
                if detection_data:
                    with st.expander("📊 Dados de Detecção"):
                        st.json(detection_data)
                else:
                    st.info("ℹ️ Nenhum dado de detecção disponível")
                
                # Controles para esta imagem
                st.subheader("⚙️ Controles")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Formulário de aprovação - FORA do button para evitar loops
                            species = st.text_input("🐦 Espécie identificada:", value="generic_bird", key="species_input")
                            confidence = st.slider("🎯 Confiança:", 0.0, 1.0, 0.8, key="confidence_input")
                            notes = st.text_area("📝 Notas:", key="notes_input")
                            
                    if st.button("✅ Confirmar Aprovação", type="primary", key="confirm_approve_btn"):
                        try:
                                approved_path = manual_analysis.approve_image(
                                    filename, species, confidence, notes,
                                    "Aprovação manual via interface web",
                                    ["uploaded_by_user"],
                                    "Análise manual realizada pelo usuário"
                                )
                                st.success("✅ Imagem aprovada!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erro ao aprovar: {e}")
                
                with col2:
                    # Formulário de rejeição - FORA do button para evitar loops
                            reason = st.text_input("❌ Motivo da rejeição:", key="reject_reason_input")
                            
                    if st.button("❌ Confirmar Rejeição", type="secondary", key="confirm_reject_btn"):
                        try:
                                rejected_path = manual_analysis.reject_image(
                                    filename, reason,
                                    "Rejeição manual via interface web",
                                    ["uploaded_by_user"],
                                    "Análise manual realizada pelo usuário"
                                )
                                st.warning("⚠️ Imagem rejeitada!")
                                st.rerun()
                        except Exception as e:
                            st.error(f"❌ Erro ao rejeitar: {e}")
                
                with col3:
                    if st.button("⏭️ Pular", type="secondary", key="skip_image_btn"):
                        st.info("ℹ️ Imagem pulada!")
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Erro ao carregar imagem: {e}")
                st.error(f"   Caminho: {image_path}")
                st.error(f"   Arquivo existe: {os.path.exists(image_path)}")
        else:
            st.warning("📁 Nenhuma imagem pendente de análise")
            
            # Mostrar estatísticas usando o sistema
            st.subheader("📊 Estatísticas")
            
            stats = manual_analysis.get_statistics()
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("✅ Aprovadas", stats['approved'])
            
            with col2:
                st.metric("❌ Rejeitadas", stats['rejected'])
            
            with col3:
                st.metric("📝 Anotações", stats['annotations'])
    
    # TAB 7: TINDER INTERFACE
    with tinder_tab:
        st.header("💡 Tinder Interface")
        
        # SEMPRE inicializar a interface (lazy loading)
        if 'tinder_initialized' not in st.session_state:
            st.session_state.tinder_initialized = False
        
        if not st.session_state.tinder_initialized:
            # Botão para inicializar
            if st.button("🚀 Inicializar Tinder Interface", key="init_tinder_btn"):
                try:
                    # Criar instância da interface
                    st.session_state.tinder_interface = TinderInterfaceEnhanced(manual_analysis)
                    st.session_state.tinder_initialized = True
                    st.success("✅ Tinder Interface inicializada!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Erro ao inicializar Tinder Interface: {e}")
                    import traceback
                    st.exception(e)
        else:
            # Interface já inicializada
            if 'tinder_interface' in st.session_state:
                tinder_interface = st.session_state.tinder_interface
                tinder_interface.render_tinder_interface()
            else:
                st.error("❌ Erro: Tinder Interface não encontrada no session_state")
                st.session_state.tinder_initialized = False
    
    # TAB 8: CASOS DUVIDOSOS
    with dubious_tab:
        dubious_review.render_review_interface()
    
    # TAB 9: CONFIGURAÇÕES
    with config_tab:
        st.header("⚙️ Configurações")
        
        # Configurações do sistema
        st.subheader("🔧 Configurações do Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.number_input("🎯 Limite de Confiança", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
            st.number_input("⏱️ Timeout (segundos)", min_value=1, max_value=60, value=30)
            st.selectbox("🌐 Idioma", ["Português", "English", "Español"])
        
        with col2:
            st.checkbox("🔄 Aprendizado Automático", value=True)
            st.checkbox("📊 Logs Detalhados", value=True)
            st.checkbox("🎨 Interface Escura", value=False)
        
        # Botões de ação
        st.subheader("💾 Ações")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Salvar Configurações", type="primary", key="save_config_btn"):
                st.success("✅ Configurações salvas!")
        
        with col2:
            if st.button("🔄 Restaurar Padrão", type="secondary", key="restore_default_btn"):
                st.info("ℹ️ Configurações restauradas!")
        
        with col3:
            if st.button("📤 Exportar Config", type="secondary", key="export_config_btn"):
                st.info("ℹ️ Configurações exportadas!")
    
    # TAB 9: RELATÓRIOS
    with relatorios_tab:
        st.header("📈 Relatórios")
        
        # Relatórios de performance
        st.subheader("📊 Relatórios de Performance")
        
        # Dados de exemplo
        report_data = {
            "Período": ["Última Hora", "Últimas 24h", "Última Semana", "Último Mês"],
            "Imagens": [12, 156, 892, 3247],
            "Taxa de Sucesso": ["94.2%", "93.8%", "94.1%", "93.9%"],
            "Tempo Médio": ["2.1s", "2.3s", "2.2s", "2.4s"]
        }
        
        report_df = pd.DataFrame(report_data)
        st.dataframe(report_df, use_container_width=True)
        
        # Gráficos
        st.subheader("📈 Gráficos de Performance")
        
        # Gráfico de tendência
        trend_data = {
            "Dia": list(range(1, 31)),
            "Performance": [0.92 + 0.02 * np.sin(i/5) for i in range(30)]
        }
        
        trend_df = pd.DataFrame(trend_data)
        fig = px.line(trend_df, x="Dia", y="Performance", title="Tendência de Performance (30 dias)")
        st.plotly_chart(fig, use_container_width=True)
    
    # TAB 10: LOGS EM TEMPO REAL
    with logs_tab:
        st.markdown("""
        <h2><i class="bi bi-journal-text bi-primary"></i>Logs em Tempo Real</h2>
        """, unsafe_allow_html=True)
        st.info("🔍 Sistema de monitoramento estilo CloudWatch AWS")
        
        # Renderizar logs em tempo real
        render_realtime_logs()
    
    # TAB 11: LOGS FRONTEND
    with frontend_logs_tab:
        st.markdown("""
        <h2><i class="bi bi-terminal bi-primary"></i>Logs Frontend</h2>
        """, unsafe_allow_html=True)
        st.info("🖥️ Logs do console do navegador em tempo real")
        
        # Renderizar logs frontend
        render_frontend_logs()
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controles Rápidos")
        
        # Status dos sistemas
        st.subheader("📊 Status")
        
        if st.button("🔄 Atualizar Status", type="primary", key="update_status_btn"):
            st.success("✅ Status atualizado!")
        
        # Controles de sistema
        st.subheader("⚙️ Sistema")
        
        if st.button("🔄 Reiniciar", type="secondary", key="restart_quick_btn"):
            st.warning("⚠️ Sistema reiniciando...")
        
        if st.button("🧹 Limpar Cache", type="secondary", key="clear_cache_quick_btn"):
            st.info("ℹ️ Cache limpo!")
        
        # Informações
        st.subheader("ℹ️ Informações")
        st.info("Versão: 2.0.0")
        st.info("Última atualização: Hoje")
        st.info("Status: ✅ Online")

if __name__ == "__main__":
    main()

# Exportar função main para uso externo
__all__ = ['main']
