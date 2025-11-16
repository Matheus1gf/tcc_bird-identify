#!/usr/bin/env python3
"""
Interface para Revisão de Casos Duvidosos
MELHORIA 2: Sistema de Priorização de Casos Duvidosos
"""

import streamlit as st
import os
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class DubiousReviewInterface:
    """Interface para revisão de casos duvidosos"""
    
    def __init__(self, dubious_dir: str = "data/manual_review/dubious"):
        """
        Inicializa interface de revisão de casos duvidosos
        
        Args:
            dubious_dir: Diretório onde casos duvidosos são armazenados
        """
        self.dubious_dir = dubious_dir
        os.makedirs(dubious_dir, exist_ok=True)
        os.makedirs(os.path.join(dubious_dir, "approved"), exist_ok=True)
        os.makedirs(os.path.join(dubious_dir, "rejected"), exist_ok=True)
        os.makedirs(os.path.join(dubious_dir, "pending"), exist_ok=True)
    
    def save_dubious_case(self, image_path: str, analysis: Dict[str, Any]) -> str:
        """
        Salva um caso duvidoso para revisão manual
        
        Args:
            image_path: Caminho da imagem
            analysis: Análise completa da imagem com informações de caso duvidoso
        
        Returns:
            Caminho do arquivo JSON salvo
        """
        try:
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            # Copiar imagem para pasta de casos duvidosos
            pending_dir = os.path.join(self.dubious_dir, "pending")
            new_image_path = os.path.join(pending_dir, filename)
            shutil.copy2(image_path, new_image_path)
            
            # Salvar metadados
            metadata = {
                'timestamp': datetime.now().isoformat(),
                'image_path': new_image_path,
                'original_path': image_path,
                'is_dubious_case': True,
                'dubious_reasons': analysis.get('dubious_reasons', []),
                'dubious_suggestion': analysis.get('dubious_suggestion', 'revisar'),
                'confidence': analysis.get('confidence', 0.0),
                'confidence_tier': analysis.get('confidence_tier', 'Muito Baixa'),
                'is_bird': analysis.get('is_bird', False),
                'species': analysis.get('species', 'Desconhecida'),
                'full_analysis': analysis
            }
            
            json_path = os.path.join(pending_dir, f"{base_name}.json")
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"[DUBIOUS_REVIEW] Caso duvidoso salvo: {filename}")
            return json_path
            
        except Exception as e:
            logger.error(f"[DUBIOUS_REVIEW] Erro ao salvar caso duvidoso: {e}")
            raise
    
    def get_pending_cases(self) -> List[Dict[str, Any]]:
        """
        Retorna lista de casos duvidosos pendentes de revisão
        
        Returns:
            Lista de casos duvidosos com metadados
        """
        pending_dir = os.path.join(self.dubious_dir, "pending")
        cases = []
        
        if not os.path.exists(pending_dir):
            return cases
        
        for file in os.listdir(pending_dir):
            if file.endswith('.json'):
                json_path = os.path.join(pending_dir, file)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        case_data = json.load(f)
                        case_data['json_path'] = json_path
                        cases.append(case_data)
                except Exception as e:
                    logger.error(f"[DUBIOUS_REVIEW] Erro ao carregar caso {file}: {e}")
        
        # Ordenar por timestamp (mais recentes primeiro)
        cases.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return cases
    
    def approve_case(self, case_data: Dict[str, Any], human_feedback: Dict[str, Any]):
        """
        Aprova um caso duvidoso
        
        Args:
            case_data: Dados do caso duvidoso
            human_feedback: Feedback humano sobre a decisão
        """
        try:
            filename = os.path.basename(case_data['image_path'])
            base_name = os.path.splitext(filename)[0]
            
            # Mover para pasta de aprovados
            approved_dir = os.path.join(self.dubious_dir, "approved")
            new_image_path = os.path.join(approved_dir, filename)
            shutil.move(case_data['image_path'], new_image_path)
            
            # Atualizar metadados
            case_data['image_path'] = new_image_path
            case_data['status'] = 'approved'
            case_data['human_feedback'] = human_feedback
            case_data['approved_at'] = datetime.now().isoformat()
            
            # Mover JSON
            json_path = case_data.get('json_path')
            if json_path and os.path.exists(json_path):
                new_json_path = os.path.join(approved_dir, f"{base_name}.json")
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(case_data, f, indent=2, ensure_ascii=False)
                os.remove(json_path)
            
            logger.info(f"[DUBIOUS_REVIEW] Caso aprovado: {filename}")
            
        except Exception as e:
            logger.error(f"[DUBIOUS_REVIEW] Erro ao aprovar caso: {e}")
            raise
    
    def reject_case(self, case_data: Dict[str, Any], human_feedback: Dict[str, Any]):
        """
        Rejeita um caso duvidoso
        
        Args:
            case_data: Dados do caso duvidoso
            human_feedback: Feedback humano sobre a decisão
        """
        try:
            filename = os.path.basename(case_data['image_path'])
            base_name = os.path.splitext(filename)[0]
            
            # Mover para pasta de rejeitados
            rejected_dir = os.path.join(self.dubious_dir, "rejected")
            new_image_path = os.path.join(rejected_dir, filename)
            shutil.move(case_data['image_path'], new_image_path)
            
            # Atualizar metadados
            case_data['image_path'] = new_image_path
            case_data['status'] = 'rejected'
            case_data['human_feedback'] = human_feedback
            case_data['rejected_at'] = datetime.now().isoformat()
            
            # Mover JSON
            json_path = case_data.get('json_path')
            if json_path and os.path.exists(json_path):
                new_json_path = os.path.join(rejected_dir, f"{base_name}.json")
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(case_data, f, indent=2, ensure_ascii=False)
                os.remove(json_path)
            
            logger.info(f"[DUBIOUS_REVIEW] Caso rejeitado: {filename}")
            
        except Exception as e:
            logger.error(f"[DUBIOUS_REVIEW] Erro ao rejeitar caso: {e}")
            raise
    
    def render_review_interface(self):
        """Renderiza interface Streamlit para revisão de casos duvidosos"""
        st.markdown("## 🔍 Revisão de Casos Duvidosos")
        st.markdown("---")
        
        # Estatísticas
        pending_cases = self.get_pending_cases()
        approved_dir = os.path.join(self.dubious_dir, "approved")
        rejected_dir = os.path.join(self.dubious_dir, "rejected")
        
        approved_count = len([f for f in os.listdir(approved_dir) if f.endswith('.json')]) if os.path.exists(approved_dir) else 0
        rejected_count = len([f for f in os.listdir(rejected_dir) if f.endswith('.json')]) if os.path.exists(rejected_dir) else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("⏳ Pendentes", len(pending_cases))
        with col2:
            st.metric("✅ Aprovados", approved_count)
        with col3:
            st.metric("❌ Rejeitados", rejected_count)
        
        st.markdown("---")
        
        if not pending_cases:
            st.info("✅ Nenhum caso duvidoso pendente de revisão!")
            return
        
        # Lista de casos pendentes
        st.markdown("### 📋 Casos Pendentes de Revisão")
        
        for i, case in enumerate(pending_cases):
            with st.expander(f"🔍 Caso #{i+1}: {os.path.basename(case.get('image_path', 'desconhecido'))}", expanded=(i == 0)):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Exibir imagem
                    image_path = case.get('image_path')
                    if image_path and os.path.exists(image_path):
                        try:
                            from PIL import Image
                            img = Image.open(image_path)
                            st.image(img, caption=os.path.basename(image_path), use_container_width=True)
                        except Exception as e:
                            st.error(f"Erro ao carregar imagem: {e}")
                    
                    # Informações do caso
                    st.markdown("#### 📊 Análise da IA")
                    st.write(f"**É pássaro:** {'✅ Sim' if case.get('is_bird') else '❌ Não'}")
                    st.write(f"**Confiança:** {case.get('confidence', 0):.2%}")
                    st.write(f"**Tier de Confiança:** {case.get('confidence_tier', 'Desconhecido')}")
                    st.write(f"**Espécie:** {case.get('species', 'Desconhecida')}")
                    
                    # Razões de dúvida
                    st.markdown("#### ⚠️ Razões de Dúvida")
                    dubious_reasons = case.get('dubious_reasons', [])
                    if dubious_reasons:
                        for reason in dubious_reasons:
                            st.warning(f"• {reason}")
                    else:
                        st.info("Nenhuma razão específica registrada")
                    
                    # Sugestão
                    suggestion = case.get('dubious_suggestion', 'revisar')
                    suggestion_icons = {
                        'aprovar': '✅',
                        'rejeitar': '❌',
                        'revisar': '🔍'
                    }
                    st.markdown(f"#### 💡 Sugestão: {suggestion_icons.get(suggestion, '❓')} {suggestion.title()}")
                
                with col2:
                    st.markdown("#### 👤 Revisão Manual")
                    
                    # Botões de ação
                    if st.button(f"✅ Aprovar", key=f"approve_{i}", type="primary"):
                        human_feedback = {
                            'is_bird': True,
                            'confidence': 1.0,
                            'reasoning': 'Aprovado após revisão manual'
                        }
                        try:
                            self.approve_case(case, human_feedback)
                            st.success("✅ Caso aprovado!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao aprovar: {e}")
                    
                    if st.button(f"❌ Rejeitar", key=f"reject_{i}"):
                        human_feedback = {
                            'is_bird': False,
                            'confidence': 1.0,
                            'reasoning': 'Rejeitado após revisão manual'
                        }
                        try:
                            self.reject_case(case, human_feedback)
                            st.success("❌ Caso rejeitado!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erro ao rejeitar: {e}")
                    
                    # Informações adicionais
                    st.markdown("---")
                    st.markdown("**📅 Data:**")
                    timestamp = case.get('timestamp', '')
                    if timestamp:
                        try:
                            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                            st.caption(dt.strftime('%d/%m/%Y %H:%M:%S'))
                        except:
                            st.caption(timestamp)
                
                st.markdown("---")

