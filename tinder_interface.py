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
    
    def approve_current_image(self, species: str, confidence: float, notes: str = ""):
        """Aprova imagem atual"""
        print(f"DEBUG - Iniciando aprovação de imagem")
        print(f"DEBUG - Pending images: {len(self.pending_images) if self.pending_images else 0}")
        print(f"DEBUG - Current index: {self.current_image_index}")
        
        if not self.pending_images or self.current_image_index >= len(self.pending_images):
            print(f"DEBUG - ERRO: Nenhuma imagem pendente ou índice inválido")
            return False
        
        current_image = self.pending_images[self.current_image_index]
        print(f"DEBUG - Imagem atual: {current_image['filename']}")
        print(f"DEBUG - Species: {species}, Confidence: {confidence}")
        
        try:
            approved_path = self.manual_analysis.approve_image(
                current_image['filename'],
                species,
                confidence,
                notes
            )
            print(f"DEBUG - Imagem aprovada com sucesso: {approved_path}")
            return True
        except Exception as e:
            print(f"DEBUG - ERRO ao aprovar imagem: {e}")
            st.error(f"Erro ao aprovar imagem: {e}")
            return False
    
    def reject_current_image(self, reason: str = ""):
        """Rejeita imagem atual"""
        if not self.pending_images or self.current_image_index >= len(self.pending_images):
            return False
        
        current_image = self.pending_images[self.current_image_index]
        try:
            rejected_path = self.manual_analysis.reject_image(
                current_image['filename'],
                reason
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
        
        # Controles estilo Tinder
        st.markdown("""
        <div style="text-align: center; margin: 30px 0;">
        """, unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
        
        with col1:
            if st.button("⬅️", key="prev_btn", help="Imagem anterior"):
                if self.previous_image():
                    st.rerun()
        
        with col2:
            # Botão de rejeição (vermelho) - SIMPLIFICADO
            if st.button("❌ REJEITAR", key="reject_btn", help="Rejeitar imagem", type="secondary"):
                if self.reject_current_image("Rejeitado pelo usuário"):
                    st.success("✅ Imagem rejeitada!")
                    # Aguardar um pouco antes de recarregar
                    import time
                    time.sleep(0.5)
                    st.rerun()
        
        with col3:
            # Botão de aprovação (verde) - SIMPLIFICADO
            if st.button("✅ APROVAR", key="approve_btn", help="Aprovar imagem", type="primary"):
                # Usar valores padrão para aprovação rápida
                species = "generic_bird"  # Espécie genérica
                confidence = 0.8  # Confiança padrão
                notes = "Aprovado pelo usuário via interface Tinder"
                
                print(f"DEBUG - Botão de aprovação clicado!")
                
                if self.approve_current_image(species, confidence, notes):
                    st.success("✅ Imagem aprovada!")
                    # Aguardar um pouco antes de recarregar
                    import time
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("❌ Falha na aprovação!")
        
        with col4:
            if st.button("➡️", key="next_btn", help="Próxima imagem"):
                if self.next_image():
                    st.rerun()
        
        with col5:
            if st.button("🔄", key="refresh_btn", help="Atualizar lista"):
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
