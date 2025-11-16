#!/usr/bin/env python3
"""
Componente Streamlit para Logs em Tempo Real
"""

import streamlit as st
import time
import json
from datetime import datetime
from typing import List, Dict, Any
from src.utils.realtime_logger import realtime_logger, LogLevel, LogEntry

class RealtimeLogViewer:
    """Visualizador de logs em tempo real para Streamlit"""
    
    def __init__(self):
        self.log_container = None
        self.auto_refresh = True
        self.refresh_interval = 1.0  # segundos
        self.max_logs = 100
        self.filter_level = None
        self.filter_module = ""
    
    def render(self):
        """Renderiza o visualizador de logs"""
        st.subheader("📊 Logs em Tempo Real")
        
        # Controles
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            self.auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
        
        with col2:
            self.refresh_interval = st.slider("⏱️ Intervalo (s)", 0.5, 5.0, 1.0, 0.5)
        
        with col3:
            level_options = ["Todos"] + [level.value for level in LogLevel]
            selected_level = st.selectbox("📊 Nível", level_options)
            self.filter_level = None if selected_level == "Todos" else LogLevel(selected_level)
        
        with col4:
            self.filter_module = st.text_input("🔍 Módulo", placeholder="Filtrar por módulo")
        
        # Botões de ação
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🗑️ Limpar Logs"):
                realtime_logger.clear_logs()
                st.success("Logs limpos!")
                st.rerun()
        
        with col2:
            if st.button("📥 Exportar"):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"logs_export_{timestamp}.json"
                realtime_logger.export_logs(filename)
                st.success(f"Logs exportados para {filename}")
        
        with col3:
            if st.button("🔄 Atualizar"):
                st.rerun()
        
        # Container para logs
        self.log_container = st.container()
        
        # Renderizar logs
        self._render_logs()
        
        # DESABILITADO: Auto-refresh causa loop infinito
        # if self.auto_refresh:
        #     time.sleep(self.refresh_interval)
        #     st.rerun() # FIXED: Comentado para prevenir loop infinito
    
    def _render_logs(self):
        """Renderiza os logs"""
        with self.log_container:
            # Obter logs filtrados
            logs = self._get_filtered_logs()
            
            if not logs:
                st.info("📝 Nenhum log encontrado")
                return
            
            # Mostrar estatísticas
            self._render_stats(logs)
            
            # Mostrar logs
            st.subheader(f"📋 Logs ({len(logs)} entradas)")
            
            # Criar tabela de logs
            for log in reversed(logs[-self.max_logs:]):  # Mostrar mais recentes primeiro
                self._render_log_entry(log)
    
    def _get_filtered_logs(self) -> List[LogEntry]:
        """Obtém logs filtrados"""
        logs = realtime_logger.get_logs(limit=self.max_logs)
        
        # Filtrar por nível
        if self.filter_level:
            logs = [log for log in logs if log.level == self.filter_level.value]
        
        # Filtrar por módulo
        if self.filter_module:
            logs = [log for log in logs if self.filter_module.lower() in log.module.lower()]
        
        return logs
    
    def _render_stats(self, logs: List[LogEntry]):
        """Renderiza estatísticas dos logs"""
        if not logs:
            return
        
        # Contar por nível
        level_counts = {}
        module_counts = {}
        
        for log in logs:
            level_counts[log.level] = level_counts.get(log.level, 0) + 1
            module_counts[log.module] = module_counts.get(log.module, 0) + 1
        
        # Mostrar estatísticas
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Total", len(logs))
        
        with col2:
            error_count = level_counts.get("ERROR", 0)
            st.metric("❌ Erros", error_count)
        
        with col3:
            warning_count = level_counts.get("WARNING", 0)
            st.metric("⚠️ Avisos", warning_count)
        
        with col4:
            info_count = level_counts.get("INFO", 0)
            st.metric("ℹ️ Info", info_count)
    
    def _render_log_entry(self, log: LogEntry):
        """Renderiza uma entrada de log"""
        # Determinar cor baseada no nível
        color_map = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🚨"
        }
        
        icon = color_map.get(log.level, "📝")
        
        # Criar container para o log
        with st.expander(f"{icon} {log.level} - {log.message[:50]}...", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Timestamp:** {log.timestamp}")
                st.write(f"**Módulo:** {log.module}")
                st.write(f"**Função:** {log.function}")
            
            with col2:
                st.write(f"**Linha:** {log.line}")
                st.write(f"**Thread:** {log.thread_id}")
                st.write(f"**Processo:** {log.process_id}")
            
            # Mensagem completa
            st.write(f"**Mensagem:** {log.message}")
            
            # Dados extras
            if log.extra_data:
                st.write("**Dados Extras:**")
                st.json(log.extra_data)

def render_realtime_logs():
    """Função para renderizar logs em tempo real"""
    viewer = RealtimeLogViewer()
    viewer.render()

# Função de conveniência para usar em outras partes do código
def log_to_streamlit(level: str, message: str, module: str = "", function: str = "", **kwargs):
    """Log que aparece no Streamlit"""
    realtime_logger.log(LogLevel(level), message, module, function, **kwargs)
    
    # Também mostrar no Streamlit
    if level == "ERROR":
        st.error(f"❌ {message}")
    elif level == "WARNING":
        st.warning(f"⚠️ {message}")
    elif level == "INFO":
        st.info(f"ℹ️ {message}")
    elif level == "SUCCESS":
        st.success(f"✅ {message}")
    else:
        st.write(f"📝 {message}")
