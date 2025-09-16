#!/usr/bin/env python3
"""
Interface estilo Tinder para aprovação/rejeição de imagens
"""

import streamlit as st
import os
from PIL import Image
from typing import Dict, Any, List
import json
from datetime import datetime

class TinderInterface:
    """Interface estilo Tinder para análise manual de imagens"""
    
    def __init__(self, manual_analysis_system):
        self.manual_analysis = manual_analysis_system
        self.current_image_index = 0
        self.pending_images = []
        
    def load_pending_images(self):
        """Carrega imagens pendentes"""
        self.pending_images = self.manual_analysis.get_pending_images()
        return len(self.pending_images)
    
    def get_current_image(self):
        """Retorna imagem atual"""
        if not self.pending_images or self.current_image_index >= len(self.pending_images):
            return None
        return self.pending_images[self.current_image_index]
    
    def approve_current_image(self, species: str, confidence: float, notes: str = "", 
                             decision_reason: str = "", visual_characteristics: List[str] = None, 
                             additional_observations: str = ""):
        """Aprova imagem atual com feedback detalhado"""
        if not self.pending_images or self.current_image_index >= len(self.pending_images):
            return False
        
        current_image = self.pending_images[self.current_image_index]
        
        try:
            approved_path = self.manual_analysis.approve_image(
                current_image['filename'],
                species,
                confidence,
                notes,
                decision_reason,
                visual_characteristics,
                additional_observations
            )
            return True
        except Exception as e:
            st.error(f"Erro ao aprovar imagem: {e}")
            return False
    
    def reject_current_image(self, reason: str = "", decision_reason: str = "", 
                           visual_characteristics: List[str] = None, additional_observations: str = ""):
        """Rejeita imagem atual com feedback detalhado"""
        if not self.pending_images or self.current_image_index >= len(self.pending_images):
            return False
        
        current_image = self.pending_images[self.current_image_index]
        try:
            rejected_path = self.manual_analysis.reject_image(
                current_image['filename'],
                reason,
                decision_reason,
                visual_characteristics,
                additional_observations
            )
            return True
        except Exception as e:
            st.error(f"Erro ao rejeitar imagem: {e}")
            return False
    
    def next_image(self):
        """Vai para próxima imagem"""
        if self.current_image_index < len(self.pending_images) - 1:
            self.current_image_index += 1
            return True
        return False
    
    def previous_image(self):
        """Vai para imagem anterior"""
        if self.current_image_index > 0:
            self.current_image_index -= 1
            return True
        return False
    
    def render_tinder_interface(self):
        """Renderiza interface estilo Tinder"""
        
        # Carregar imagens pendentes
        total_images = self.load_pending_images()
        
        if total_images == 0:
            st.success("🎉 Nenhuma imagem pendente!")
            st.info("Todas as imagens foram analisadas. Novas imagens aparecerão aqui quando o sistema detectar pássaros não reconhecidos.")
            return
        
        # Atualizar índice se necessário
        if self.current_image_index >= total_images:
            self.current_image_index = 0
        
        current_image = self.get_current_image()
        if not current_image:
            st.error("Erro ao carregar imagem atual")
            return
        
        # Header com progresso
        st.markdown(f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h3>Análise Manual - Estilo Tinder</h3>
            <p>Imagem {self.current_image_index + 1} de {total_images}</p>
            <div style="background-color: #f0f0f0; height: 10px; border-radius: 5px; margin: 10px 0;">
                <div style="background-color: #4CAF50; height: 100%; width: {(self.current_image_index + 1) / total_images * 100}%; border-radius: 5px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Card principal da imagem
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Container do card
            st.markdown("""
            <div style="
                background: white;
                border-radius: 20px;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                padding: 20px;
                margin: 20px 0;
                text-align: center;
            ">
            """, unsafe_allow_html=True)
            
            # Imagem
            try:
                image = Image.open(current_image['image_path'])
                st.image(image, caption=current_image['filename'], use_container_width=True)
            except Exception as e:
                st.error(f"Erro ao carregar imagem: {e}")
            
            # Informações da detecção
            detection_data = current_image['detection_data']
            st.markdown("### 📊 Dados de Detecção")
            
            if detection_data.get('yolo_detections'):
                st.write("**Detecções YOLO:**")
                for det in detection_data['yolo_detections']:
                    st.write(f"• {det.get('class', 'N/A')}: {det.get('confidence', 0):.2%}")
            else:
                st.write("Nenhuma detecção YOLO")
            
            st.write(f"**Tipo de Análise:** {detection_data.get('analysis_type', 'N/A')}")
            st.write(f"**Timestamp:** {detection_data.get('timestamp', 'N/A')}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Formulário de Feedback Detalhado
        st.markdown("### 🧠 Feedback Detalhado para o Machine Learning")
        st.info("💡 Forneça informações detalhadas para ajudar a IA a aprender com sua decisão!")
        
        with st.form("feedback_form", clear_on_submit=True):
            # Seção de Decisão
            st.markdown("#### 📋 Sua Decisão")
            
            col1, col2 = st.columns(2)
            
            with col1:
                decision = st.radio(
                    "O que você decidiu?",
                    ["✅ APROVAR como pássaro", "❌ REJEITAR (não é pássaro)"],
                    key="decision_radio"
                )
            
            with col2:
                if decision == "✅ APROVAR como pássaro":
                    species = st.text_input(
                        "Nome da espécie do pássaro:",
                        placeholder="Ex: Bem-te-vi, Sabiá, Beija-flor...",
                        key="species_input"
                    )
                    confidence = st.slider(
                        "Confiança na identificação:",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.8,
                        step=0.1,
                        key="confidence_slider"
                    )
                else:
                    species = "not_a_bird"
                    confidence = 0.0
            
            # Seção de Características Visuais
            st.markdown("#### 🔍 Características Visuais Detectadas")
            st.write("Marque as características que você consegue identificar na imagem:")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                visual_characteristics = []
                if st.checkbox("🪶 Asas visíveis", key="wings_check"):
                    visual_characteristics.append("asas")
                if st.checkbox("🐦 Bico visível", key="beak_check"):
                    visual_characteristics.append("bico")
                if st.checkbox("👁️ Olhos visíveis", key="eyes_check"):
                    visual_characteristics.append("olhos")
                if st.checkbox("🦵 Pernas/Patas visíveis", key="legs_check"):
                    visual_characteristics.append("pernas")
            
            with col2:
                if st.checkbox("🪶 Penas visíveis", key="feathers_check"):
                    visual_characteristics.append("penas")
                if st.checkbox("🎨 Padrão de cores distintivo", key="pattern_check"):
                    visual_characteristics.append("padrao_cores")
                if st.checkbox("🧠 Intuição IA detectou características", key="intuition_check"):
                    visual_characteristics.append("intuicao_ia")
                if st.checkbox("🌿 Em ambiente natural", key="habitat_check"):
                    visual_characteristics.append("habitat_natural")
            
            with col3:
                if st.checkbox("✈️ Posição de voo", key="flight_check"):
                    visual_characteristics.append("posicao_voo")
                if st.checkbox("🌳 Pousado em galho", key="perched_check"):
                    visual_characteristics.append("pousado_galho")
                if st.checkbox("🍽️ Alimentando-se", key="feeding_check"):
                    visual_characteristics.append("alimentando")
                if st.checkbox("🎵 Cantando", key="singing_check"):
                    visual_characteristics.append("cantando")
            
            # Seção de Motivação da Decisão
            st.markdown("#### 🤔 Por que você tomou essa decisão?")
            
            decision_reason = st.text_area(
                "Explique o que te fez decidir por esta opção:",
                placeholder="Ex: Posso ver claramente as asas e o bico característico do bem-te-vi...",
                height=100,
                key="decision_reason_text"
            )
            
            # Seção de Observações Adicionais
            st.markdown("#### 📝 Observações Adicionais")
            
            additional_observations = st.text_area(
                "Outras informações que podem ajudar a IA a aprender:",
                placeholder="Ex: A iluminação estava boa, o pássaro estava em foco, posso ver detalhes das penas...",
                height=100,
                key="additional_observations_text"
            )
            
            # Seção de Notas
            notes = st.text_input(
                "Notas gerais (opcional):",
                placeholder="Qualquer informação adicional...",
                key="notes_input"
            )
            
            # Botões de Ação
            st.markdown("---")
            
            col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
            
            with col1:
                if st.form_submit_button("⬅️ Anterior", help="Imagem anterior"):
                    if self.previous_image():
                        st.rerun()
            
            with col2:
                if st.form_submit_button("❌ REJEITAR", help="Rejeitar com feedback", type="secondary"):
                    if decision == "❌ REJEITAR (não é pássaro)":
                        if self.reject_current_image(
                            "Rejeitado pelo usuário",
                            decision_reason,
                            visual_characteristics,
                            additional_observations
                        ):
                            st.success("✅ Imagem rejeitada com feedback!")
                            import time
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Falha na rejeição!")
                    else:
                        st.warning("⚠️ Selecione 'REJEITAR' na decisão primeiro!")
            
            with col3:
                if st.form_submit_button("✅ APROVAR", help="Aprovar com feedback", type="primary"):
                    if decision == "✅ APROVAR como pássaro" and species and species != "not_a_bird":
                        if self.approve_current_image(
                            species,
                            confidence,
                            notes,
                            decision_reason,
                            visual_characteristics,
                            additional_observations
                        ):
                            st.success("✅ Imagem aprovada com feedback!")
                            import time
                            time.sleep(0.5)
                            st.rerun()
                        else:
                            st.error("❌ Falha na aprovação!")
                    else:
                        st.warning("⚠️ Preencha o nome da espécie primeiro!")
            
            with col4:
                if st.form_submit_button("➡️ Próxima", help="Próxima imagem"):
                    if self.next_image():
                        st.rerun()
            
            with col5:
                if st.form_submit_button("🔄 Atualizar", help="Atualizar lista"):
                    st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Estatísticas
        stats = self.manual_analysis.get_statistics()
        
        st.markdown("---")
        st.markdown("### 📈 Estatísticas")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Pendentes", stats['pending'])
        with col2:
            st.metric("Aprovadas", stats['approved'])
        with col3:
            st.metric("Rejeitadas", stats['rejected'])
        with col4:
            st.metric("Anotações", stats['annotations'])
        
        # Instruções
        st.markdown("---")
        st.markdown("""
        ### 📋 Instruções
        
        **Como usar a interface estilo Tinder:**
        
        1. **❌ Rejeitar**: Clique no X vermelho se a imagem não for útil para treinamento
        2. **✅ Aprovar**: Clique no ✓ verde se a imagem for um pássaro válido
        3. **⬅️➡️ Navegar**: Use as setas para navegar entre imagens
        4. **🔄 Atualizar**: Clique no botão de atualizar para recarregar a lista
        
        **Dicas:**
        - Aprove apenas imagens claras de pássaros
        - Rejeite imagens borradas, sem pássaros ou com qualidade ruim
        - Seja consistente nas suas classificações
        """)
