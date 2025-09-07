#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Aplicação Streamlit Simplificada para Teste de Botões
"""

import streamlit as st
import os
from datetime import datetime

# Configuração da página
st.set_page_config(
    page_title="Sistema de Análise de Pássaros - Teste",
    page_icon="🐦",
    layout="wide"
)

st.title("🐦 Sistema de Análise de Pássaros - Teste Simplificado")

# Inicializar variáveis de sessão
if 'image_uploaded' not in st.session_state:
    st.session_state.image_uploaded = False
if 'temp_path' not in st.session_state:
    st.session_state.temp_path = None

# Upload de imagem
st.header("📁 Upload de Imagem")
uploaded_file = st.file_uploader("Escolha uma imagem", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    st.session_state.image_uploaded = True
    
    # Salvar imagem temporariamente
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.session_state.temp_path = temp_path
    
    st.success(f"✅ Imagem carregada: {uploaded_file.name}")
    st.image(uploaded_file, caption="Imagem carregada", use_column_width=True)

# Seção de teste de botões
st.header("🧪 Teste de Botões")

if st.session_state.image_uploaded:
    st.info("Imagem carregada - botões disponíveis")
    
    # Botão 1: Teste simples
    if st.button("TESTE SIMPLES", key="test_simple"):
        st.success("✅ Botão simples funcionou!")
        st.write("Se você vê esta mensagem, o botão está funcionando!")
    
    # Botão 2: Teste com arquivo
    if st.button("TESTE COM ARQUIVO", key="test_file"):
        if st.session_state.temp_path and os.path.exists(st.session_state.temp_path):
            file_size = os.path.getsize(st.session_state.temp_path)
            st.success(f"✅ Arquivo encontrado: {st.session_state.temp_path}")
            st.write(f"📁 Tamanho: {file_size} bytes")
        else:
            st.error("❌ Arquivo temporário não encontrado!")
    
    # Botão 3: Simular análise manual
    if st.button("SIMULAR ANÁLISE MANUAL", key="simulate_manual"):
        try:
            # Simular processo de análise manual
            st.write("🔄 Simulando análise manual...")
            
            if st.session_state.temp_path and os.path.exists(st.session_state.temp_path):
                # Criar diretório se não existir
                os.makedirs("manual_analysis/pending", exist_ok=True)
                
                # Copiar arquivo
                import shutil
                pending_path = f"manual_analysis/pending/manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(st.session_state.temp_path)}"
                shutil.copy2(st.session_state.temp_path, pending_path)
                
                if os.path.exists(pending_path):
                    st.success("✅ Imagem salva para análise manual!")
                    st.write(f"📁 Salva em: {pending_path}")
                    
                    # Remover arquivo temporário
                    os.remove(st.session_state.temp_path)
                    st.session_state.temp_path = None
                    st.session_state.image_uploaded = False
                    st.write("🗑️ Arquivo temporário removido")
                else:
                    st.error("❌ Falha ao salvar arquivo!")
            else:
                st.error("❌ Arquivo temporário não encontrado!")
                
        except Exception as e:
            st.error(f"❌ Erro: {e}")
    
    # Botão 4: Ver arquivos pendentes
    if st.button("VER ARQUIVOS PENDENTES", key="view_pending"):
        pending_dir = "manual_analysis/pending"
        if os.path.exists(pending_dir):
            files = os.listdir(pending_dir)
            if files:
                st.write(f"📁 Encontrados {len(files)} arquivos:")
                for file in files:
                    file_path = os.path.join(pending_dir, file)
                    file_size = os.path.getsize(file_path)
                    st.write(f"  • {file} ({file_size} bytes)")
            else:
                st.write("📭 Nenhum arquivo pendente")
        else:
            st.write("❌ Diretório não encontrado")

else:
    st.warning("⚠️ Faça upload de uma imagem primeiro")

# Informações de debug
st.header("🔍 Informações de Debug")
st.write(f"**Imagem carregada**: {st.session_state.image_uploaded}")
st.write(f"**Arquivo temporário**: {st.session_state.temp_path}")
if st.session_state.temp_path:
    st.write(f"**Arquivo existe**: {os.path.exists(st.session_state.temp_path)}")

# Log de sessão
st.header("📋 Log de Sessão")
if st.button("LIMPAR LOG", key="clear_log"):
    st.session_state.log_messages = []

if 'log_messages' not in st.session_state:
    st.session_state.log_messages = []

# Adicionar mensagem ao log
if st.button("ADICIONAR LOG", key="add_log"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = f"[{timestamp}] Teste de log adicionado"
    st.session_state.log_messages.append(message)

# Exibir log
for message in st.session_state.log_messages:
    st.write(message)
