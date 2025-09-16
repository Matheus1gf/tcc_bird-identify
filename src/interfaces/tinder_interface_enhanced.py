#!/usr/bin/env python3
"""
Interface Tinder Melhorada para Análise Manual de Pássaros
Sistema de aprendizado contínuo baseado em feedback humano
"""

import streamlit as st
import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from PIL import Image
import cv2
import numpy as np

class TinderInterfaceEnhanced:
    """Interface Tinder melhorada para análise manual de pássaros"""
    
    def __init__(self, manual_analysis_system):
        self.manual_analysis = manual_analysis_system
        self.session_data = {
            'current_image': None,
            'current_analysis': None,
            'feedback_history': [],
            'learning_events': []
        }
        
        # Configurar CSS personalizado
        self._setup_custom_css()
    
    def _setup_custom_css(self):
        """Configura CSS personalizado para interface Tinder"""
        st.markdown("""
        <style>
        .tinder-container {
            max-width: 500px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }
        
        .tinder-card {
            background: white;
            border-radius: 20px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            text-align: center;
        }
        
        .tinder-image {
            width: 100%;
            max-width: 400px;
            height: 300px;
            object-fit: cover;
            border-radius: 15px;
            margin: 10px 0;
        }
        
        .tinder-buttons {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        
        .tinder-button {
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            font-size: 18px;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .tinder-button.reject {
            background: #ff4757;
            color: white;
        }
        
        .tinder-button.reject:hover {
            background: #ff3742;
            transform: scale(1.05);
        }
        
        .tinder-button.approve {
            background: #2ed573;
            color: white;
        }
        
        .tinder-button.approve:hover {
            background: #26d065;
            transform: scale(1.05);
        }
        
        .tinder-button.learn {
            background: #3742fa;
            color: white;
        }
        
        .tinder-button.learn:hover {
            background: #2f3542;
            transform: scale(1.05);
        }
        
        .analysis-info {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #667eea;
        }
        
        .characteristics-list {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 10px 0;
        }
        
        .characteristic-tag {
            background: #667eea;
            color: white;
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
        }
        
        .learning-feedback {
            background: #e8f5e8;
            padding: 15px;
            border-radius: 10px;
            margin: 10px 0;
            border-left: 4px solid #2ed573;
        }
        
        .stats-container {
            display: flex;
            justify-content: space-around;
            margin: 20px 0;
        }
        
        .stat-item {
            text-align: center;
            background: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .stat-number {
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }
        
        .stat-label {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def render_tinder_interface(self):
        """Renderiza a interface Tinder principal"""
        st.markdown("""
        <div class="tinder-container">
            <h1 style="text-align: center; color: white; margin-bottom: 30px;">
                🐦 Análise Manual de Pássaros
            </h1>
            <p style="text-align: center; color: white; margin-bottom: 30px;">
                Ajude a IA a aprender como uma criança descobrindo pássaros!
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Verificar se há imagens para análise
        pending_images = self._get_pending_images()
        
        if not pending_images:
            self._render_no_images_message()
            return
        
        # Renderizar estatísticas
        self._render_statistics()
        
        # Renderizar imagem atual
        current_image = self._get_current_image()
        if current_image:
            self._render_image_card(current_image)
        else:
            self._render_image_selection(pending_images)
    
    def _get_pending_images(self) -> List[str]:
        """Obtém lista de imagens pendentes para análise"""
        pending_dir = "manual_analysis/pending"
        if not os.path.exists(pending_dir):
            os.makedirs(pending_dir)
            return []
        
        image_files = []
        for file in os.listdir(pending_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_files.append(os.path.join(pending_dir, file))
        
        return image_files
    
    def _get_current_image(self) -> Optional[str]:
        """Obtém a imagem atual sendo analisada"""
        return st.session_state.get('current_tinder_image')
    
    def _render_no_images_message(self):
        """Renderiza mensagem quando não há imagens para análise"""
        st.markdown("""
        <div class="tinder-card">
            <h3>🎯 Nenhuma imagem pendente</h3>
            <p>Não há imagens aguardando análise manual no momento.</p>
            <p>Faça upload de uma imagem na aba principal para começar a análise!</p>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_statistics(self):
        """Renderiza estatísticas de aprendizado"""
        stats = self._get_learning_statistics()
        
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-item">
                <div class="stat-number">{stats['total_analyzed']}</div>
                <div class="stat-label">Imagens Analisadas</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{stats['birds_identified']}</div>
                <div class="stat-label">Pássaros Identificados</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{stats['species_learned']}</div>
                <div class="stat-label">Espécies Aprendidas</div>
            </div>
            <div class="stat-item">
                <div class="stat-number">{stats['learning_events']}</div>
                <div class="stat-label">Eventos de Aprendizado</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _get_learning_statistics(self) -> Dict[str, int]:
        """Obtém estatísticas de aprendizado"""
        # Contar imagens analisadas
        approved_dir = "manual_analysis/approved"
        rejected_dir = "manual_analysis/rejected"
        
        total_analyzed = 0
        birds_identified = 0
        species_learned = 0
        learning_events = 0
        
        if os.path.exists(approved_dir):
            approved_files = [f for f in os.listdir(approved_dir) if f.endswith('.json')]
            total_analyzed += len(approved_files)
            birds_identified += len(approved_files)
            
            # Contar espécies únicas
            species_set = set()
            for file in approved_files:
                try:
                    with open(os.path.join(approved_dir, file), 'r') as f:
                        data = json.load(f)
                        if 'species' in data:
                            species_set.add(data['species'])
                except:
                    pass
            species_learned = len(species_set)
        
        if os.path.exists(rejected_dir):
            rejected_files = [f for f in os.listdir(rejected_dir) if f.endswith('.json')]
            total_analyzed += len(rejected_files)
        
        # Contar eventos de aprendizado
        learning_dir = "learning_data"
        if os.path.exists(learning_dir):
            for subdir in os.listdir(learning_dir):
                subdir_path = os.path.join(learning_dir, subdir)
                if os.path.isdir(subdir_path):
                    learning_events += len([f for f in os.listdir(subdir_path) if f.endswith('.json')])
        
        return {
            'total_analyzed': total_analyzed,
            'birds_identified': birds_identified,
            'species_learned': species_identified,
            'learning_events': learning_events
        }
    
    def _render_image_selection(self, pending_images: List[str]):
        """Renderiza seleção de imagem"""
        st.markdown("""
        <div class="tinder-card">
            <h3>📸 Selecione uma imagem para análise</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for i, image_path in enumerate(pending_images[:5]):  # Mostrar apenas 5 primeiras
            col1, col2 = st.columns([3, 1])
            
            with col1:
                try:
                    image = Image.open(image_path)
                    st.image(image, caption=os.path.basename(image_path), use_column_width=True)
                except Exception as e:
                    st.error(f"Erro ao carregar imagem: {e}")
            
            with col2:
                if st.button(f"Analisar", key=f"select_{i}"):
                    st.session_state['current_tinder_image'] = image_path
                    st.rerun()
    
    def _render_image_card(self, image_path: str):
        """Renderiza card da imagem atual"""
        try:
            # Carregar imagem
            image = Image.open(image_path)
            
            # Obter análise da IA
            analysis = self._get_ai_analysis(image_path)
            
            st.markdown(f"""
            <div class="tinder-card">
                <h3>🐦 Análise da IA</h3>
                <img src="data:image/jpeg;base64,{self._image_to_base64(image)}" class="tinder-image">
            </div>
            """, unsafe_allow_html=True)
            
            # Mostrar análise da IA
            if analysis:
                self._render_ai_analysis(analysis)
            
            # Botões de ação
            self._render_action_buttons(image_path, analysis)
            
        except Exception as e:
            st.error(f"Erro ao renderizar imagem: {e}")
    
    def _get_ai_analysis(self, image_path: str) -> Optional[Dict[str, Any]]:
        """Obtém análise da IA para a imagem"""
        try:
            # Simular análise da IA (substituir por análise real)
            return {
                'confidence': 0.75,
                'species': 'Pássaro de espécie desconhecida',
                'color': 'brown',
                'characteristics': ['has_eyes', 'has_feathers', 'bird_body_shape'],
                'reasoning': 'Detectadas características visuais interessantes'
            }
        except Exception as e:
            st.error(f"Erro na análise da IA: {e}")
            return None
    
    def _render_ai_analysis(self, analysis: Dict[str, Any]):
        """Renderiza análise da IA"""
        st.markdown(f"""
        <div class="analysis-info">
            <h4>🤖 Análise da IA</h4>
            <p><strong>Confiança:</strong> {analysis['confidence']:.1%}</p>
            <p><strong>Espécie:</strong> {analysis['species']}</p>
            <p><strong>Cor:</strong> {analysis['color']}</p>
            <p><strong>Raciocínio:</strong> {analysis['reasoning']}</p>
            
            <div class="characteristics-list">
                {''.join([f'<span class="characteristic-tag">{char}</span>' for char in analysis['characteristics']])}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    def _render_action_buttons(self, image_path: str, analysis: Dict[str, Any]):
        """Renderiza botões de ação"""
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("❌ Não é Pássaro", key="reject"):
                self._handle_rejection(image_path, analysis)
        
        with col2:
            if st.button("✅ É Pássaro", key="approve"):
                self._handle_approval(image_path, analysis)
        
        with col3:
            if st.button("🧠 Ensinar IA", key="learn"):
                self._handle_learning(image_path, analysis)
    
    def _handle_rejection(self, image_path: str, analysis: Dict[str, Any]):
        """Processa rejeição da imagem"""
        try:
            # Mover para pasta de rejeitados
            rejected_dir = "manual_analysis/rejected"
            os.makedirs(rejected_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            new_path = os.path.join(rejected_dir, filename)
            
            # Copiar imagem
            import shutil
            shutil.copy2(image_path, new_path)
            
            # Salvar feedback
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'decision': 'rejected',
                'reason': 'Não é um pássaro',
                'ai_analysis': analysis,
                'human_feedback': {
                    'is_bird': False,
                    'confidence': 1.0,
                    'reasoning': 'Imagem rejeitada pelo usuário'
                }
            }
            
            feedback_file = os.path.join(rejected_dir, f"{filename}.json")
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            # Remover da pasta pendente
            os.remove(image_path)
            
            # Limpar imagem atual
            if 'current_tinder_image' in st.session_state:
                del st.session_state['current_tinder_image']
            
            st.success("✅ Imagem rejeitada e feedback salvo!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro ao processar rejeição: {e}")
    
    def _handle_approval(self, image_path: str, analysis: Dict[str, Any]):
        """Processa aprovação da imagem"""
        try:
            # Mover para pasta de aprovados
            approved_dir = "manual_analysis/approved"
            os.makedirs(approved_dir, exist_ok=True)
            
            filename = os.path.basename(image_path)
            new_path = os.path.join(approved_dir, filename)
            
            # Copiar imagem
            import shutil
            shutil.copy2(image_path, new_path)
            
            # Salvar feedback
            feedback_data = {
                'timestamp': datetime.now().isoformat(),
                'decision': 'approved',
                'reason': 'É um pássaro',
                'ai_analysis': analysis,
                'human_feedback': {
                    'is_bird': True,
                    'confidence': 1.0,
                    'reasoning': 'Imagem aprovada pelo usuário'
                }
            }
            
            feedback_file = os.path.join(approved_dir, f"{filename}.json")
            with open(feedback_file, 'w') as f:
                json.dump(feedback_data, f, indent=2)
            
            # Remover da pasta pendente
            os.remove(image_path)
            
            # Limpar imagem atual
            if 'current_tinder_image' in st.session_state:
                del st.session_state['current_tinder_image']
            
            st.success("✅ Imagem aprovada e feedback salvo!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro ao processar aprovação: {e}")
    
    def _handle_learning(self, image_path: str, analysis: Dict[str, Any]):
        """Processa aprendizado da IA"""
        try:
            # Mostrar formulário de aprendizado
            st.markdown("""
            <div class="learning-feedback">
                <h4>🧠 Ensinar a IA</h4>
                <p>Ajude a IA a aprender com esta imagem!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Formulário de feedback
            with st.form("learning_form"):
                st.write("**Informações sobre o pássaro:**")
                
                is_bird = st.radio("É um pássaro?", ["Sim", "Não"], key="is_bird")
                
                if is_bird == "Sim":
                    species = st.text_input("Espécie (se souber):", key="species")
                    color = st.selectbox("Cor predominante:", 
                                       ["brown", "black", "white", "red", "blue", "yellow", "green"], 
                                       key="color")
                    
                    st.write("**Características visíveis:**")
                    has_eyes = st.checkbox("Tem olhos", key="has_eyes")
                    has_wings = st.checkbox("Tem asas", key="has_wings")
                    has_beak = st.checkbox("Tem bico", key="has_beak")
                    has_feathers = st.checkbox("Tem penas", key="has_feathers")
                    has_claws = st.checkbox("Tem garras", key="has_claws")
                    
                    reasoning = st.text_area("Por que você sabe que é um pássaro?", key="reasoning")
                else:
                    reasoning = st.text_area("Por que não é um pássaro?", key="reasoning")
                
                submitted = st.form_submit_button("💾 Salvar Aprendizado")
                
                if submitted:
                    self._save_learning_feedback(image_path, analysis, {
                        'is_bird': is_bird == "Sim",
                        'species': species if is_bird == "Sim" else None,
                        'color': color if is_bird == "Sim" else None,
                        'characteristics': {
                            'has_eyes': has_eyes if is_bird == "Sim" else False,
                            'has_wings': has_wings if is_bird == "Sim" else False,
                            'has_beak': has_beak if is_bird == "Sim" else False,
                            'has_feathers': has_feathers if is_bird == "Sim" else False,
                            'has_claws': has_claws if is_bird == "Sim" else False
                        },
                        'reasoning': reasoning
                    })
            
        except Exception as e:
            st.error(f"Erro no aprendizado: {e}")
    
    def _save_learning_feedback(self, image_path: str, analysis: Dict[str, Any], feedback: Dict[str, Any]):
        """Salva feedback de aprendizado"""
        try:
            # Criar diretório de aprendizado
            learning_dir = "learning_data/human_feedback"
            os.makedirs(learning_dir, exist_ok=True)
            
            # Salvar dados de aprendizado
            learning_data = {
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path,
                'ai_analysis': analysis,
                'human_feedback': feedback,
                'learning_type': 'human_feedback'
            }
            
            filename = f"learning_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            learning_file = os.path.join(learning_dir, filename)
            
            with open(learning_file, 'w') as f:
                json.dump(learning_data, f, indent=2)
            
            # Mover imagem para pasta apropriada
            if feedback['is_bird']:
                target_dir = "manual_analysis/approved"
            else:
                target_dir = "manual_analysis/rejected"
            
            os.makedirs(target_dir, exist_ok=True)
            
            import shutil
            filename = os.path.basename(image_path)
            new_path = os.path.join(target_dir, filename)
            shutil.copy2(image_path, new_path)
            
            # Remover da pasta pendente
            os.remove(image_path)
            
            # Limpar imagem atual
            if 'current_tinder_image' in st.session_state:
                del st.session_state['current_tinder_image']
            
            st.success("🧠 Aprendizado salvo com sucesso!")
            st.rerun()
            
        except Exception as e:
            st.error(f"Erro ao salvar aprendizado: {e}")
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Converte imagem para base64"""
        import base64
        import io
        
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return img_str
    
    def load_pending_images(self) -> int:
        """Carrega e retorna o número de imagens pendentes de análise"""
        try:
            pending_dir = "manual_analysis/pending"
            if not os.path.exists(pending_dir):
                return 0
            
            # Contar arquivos de imagem pendentes
            image_files = []
            for file in os.listdir(pending_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(file)
            
            return len(image_files)
            
        except Exception as e:
            st.error(f"Erro ao carregar imagens pendentes: {e}")
            return 0
