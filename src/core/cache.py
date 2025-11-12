#!/usr/bin/env python3
"""
Sistema de Cache de Reconhecimento de Imagens
Evita reprocessamento de imagens já analisadas e aprovadas
"""

import os
import json
import hashlib
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class ImageRecognitionCache:
    """Cache para reconhecimento de imagens já analisadas"""
    
    def __init__(self, cache_file: str = "./image_recognition_cache.json"):
        self.cache_file = cache_file
        self.cache_data = self._load_cache()
        
    def _load_cache(self) -> Dict:
        """Carrega cache do arquivo"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Erro ao carregar cache: {e}")
                return {"images": {}, "species_database": {}}
        return {"images": {}, "species_database": {}}
    
    def _save_cache(self):
        """Salva cache no arquivo"""
        try:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao salvar cache: {e}")
    
    def _calculate_image_hash(self, image_path: str) -> str:
        """Calcula hash da imagem para identificação única"""
        try:
            # Ler imagem e calcular hash
            image = cv2.imread(image_path)
            if image is None:
                return ""
            
            # Converter para escala de cinza e redimensionar para hash consistente
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, (64, 64))
            
            # Calcular hash
            image_bytes = resized.tobytes()
            return hashlib.md5(image_bytes).hexdigest()
            
        except Exception as e:
            logger.error(f"Erro ao calcular hash da imagem {image_path}: {e}")
            return ""
    
    def _calculate_similarity(self, image_path: str, cached_hash: str) -> float:
        """Calcula similaridade entre imagem atual e cache"""
        try:
            current_hash = self._calculate_image_hash(image_path)
            if not current_hash or not cached_hash:
                return 0.0
            
            # Comparação simples de hash (pode ser melhorada com histograma)
            if current_hash == cached_hash:
                return 1.0
            
            # Calcular similaridade baseada em características visuais
            image = cv2.imread(image_path)
            if image is None:
                return 0.0
            
            # Extrair características básicas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Normalizar histograma
            hist = hist.flatten() / hist.sum()
            
            # Comparar com características salvas (se disponível)
            # Por enquanto, retornar 0.5 para imagens diferentes mas similares
            return 0.5
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {e}")
            return 0.0
    
    def is_image_recognized(self, image_path: str, similarity_threshold: float = 0.8) -> Optional[Dict]:
        """
        Verifica se a imagem já foi reconhecida anteriormente
        
        Args:
            image_path: Caminho para a imagem
            similarity_threshold: Limiar de similaridade (0.0 a 1.0)
            
        Returns:
            Dict com informações do reconhecimento ou None se não reconhecida
        """
        if not os.path.exists(image_path):
            return None
        
        current_hash = self._calculate_image_hash(image_path)
        if not current_hash:
            return None
        
        # Verificar cache por hash exato
        if current_hash in self.cache_data["images"]:
            cached_info = self.cache_data["images"][current_hash]
            logger.info(f"[ATUALIZACAO] Imagem reconhecida por hash exato: {os.path.basename(image_path)}")
            return cached_info
        
        # Verificar por similaridade
        for cached_hash, cached_info in self.cache_data["images"].items():
            similarity = self._calculate_similarity(image_path, cached_hash)
            if similarity >= similarity_threshold:
                logger.info(f"[ATUALIZACAO] Imagem reconhecida por similaridade ({similarity:.2f}): {os.path.basename(image_path)}")
                return cached_info
        
        return None
    
    def is_image_rejected(self, image_path: str, similarity_threshold: float = 0.8) -> Optional[Dict]:
        """
        Verifica se a imagem (ou similar) já foi rejeitada anteriormente
        
        Args:
            image_path: Caminho para a imagem
            similarity_threshold: Limiar de similaridade (0.0 a 1.0)
            
        Returns:
            Dict com informações da rejeição ou None se não rejeitada
        """
        if not os.path.exists(image_path):
            return None
        
        current_hash = self._calculate_image_hash(image_path)
        if not current_hash:
            return None
        
        # Verificar pasta de rejeitados
        rejected_dir = "data/manual_analysis/rejected"
        if not os.path.exists(rejected_dir):
            return None
        
        # ESTRATÉGIA MELHORADA: Comparar hash com TODAS as imagens rejeitadas primeiro
        # Isso garante detecção mesmo quando imagens têm nomes diferentes
        for filename in os.listdir(rejected_dir):
            # Processar apenas imagens, não JSONs
            if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            
            image_path_in_rejected = os.path.join(rejected_dir, filename)
            if not os.path.exists(image_path_in_rejected):
                continue
            
            # Calcular hash da imagem rejeitada
            rejected_hash = self._calculate_image_hash(image_path_in_rejected)
            if not rejected_hash:
                continue
            
            # Comparar hash exato
            if rejected_hash == current_hash:
                # Encontrar JSON correspondente se existir
                json_path = os.path.join(rejected_dir, f"{filename}.json")
                rejection_data = {}
                human_feedback = {'is_bird': False, 'confidence': 1.0, 'reasoning': 'Imagem rejeitada pelo usuário'}
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            rejection_data = json.load(f)
                            human_feedback = rejection_data.get('human_feedback', human_feedback)
                    except:
                        pass
                
                logger.info(f"[REJEIÇÃO] Imagem já foi rejeitada (hash exato): {os.path.basename(image_path)}")
                return {
                    'rejected': True,
                    'reason': rejection_data.get('reason', 'Não é um pássaro'),
                    'human_feedback': human_feedback,
                    'timestamp': rejection_data.get('timestamp', ''),
                    'confidence': human_feedback.get('confidence', 1.0),
                    'reasoning': human_feedback.get('reasoning', 'Imagem rejeitada pelo usuário')
                }
            
            # Verificar por similaridade
            similarity = self._calculate_similarity(image_path, rejected_hash)
            if similarity >= similarity_threshold:
                # Encontrar JSON correspondente
                json_path = os.path.join(rejected_dir, f"{filename}.json")
                rejection_data = {}
                human_feedback = {'is_bird': False, 'confidence': 1.0, 'reasoning': 'Imagem rejeitada pelo usuário'}
                
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            rejection_data = json.load(f)
                            human_feedback = rejection_data.get('human_feedback', human_feedback)
                    except:
                        pass
                
                logger.info(f"[REJEIÇÃO] Imagem similar já foi rejeitada ({similarity:.2f}): {os.path.basename(image_path)}")
                return {
                    'rejected': True,
                    'reason': rejection_data.get('reason', 'Não é um pássaro'),
                    'human_feedback': human_feedback,
                    'timestamp': rejection_data.get('timestamp', ''),
                    'confidence': human_feedback.get('confidence', 1.0),
                    'reasoning': human_feedback.get('reasoning', 'Imagem rejeitada pelo usuário'),
                    'similarity': similarity
                }
        
        return None
    
    def add_rejection_to_cache(self, image_path: str, rejection_data: Dict):
        """
        Adiciona rejeição ao cache para consulta rápida futura
        
        Args:
            image_path: Caminho para a imagem rejeitada
            rejection_data: Dados da rejeição
        """
        if not os.path.exists(image_path):
            return
        
        image_hash = self._calculate_image_hash(image_path)
        if not image_hash:
            return
        
        # Adicionar hash negativo ao cache (para indicar rejeição)
        self.cache_data["images"][f"rejected_{image_hash}"] = {
            "image_path": image_path,
            "rejected": True,
            "species": "Não é um pássaro",
            "confidence": 0.0,
            "rejection_data": rejection_data,
            "timestamp": rejection_data.get('timestamp', datetime.now().isoformat()),
            "recognition_type": "rejected_by_human"
        }
        
        self._save_cache()
        logger.info(f"[CACHE] Rejeição adicionada ao cache: {os.path.basename(image_path)}")
    
    def add_recognized_image(self, image_path: str, species: str, confidence: float, 
                           analysis_data: Dict, notes: str = ""):
        """
        Adiciona imagem reconhecida ao cache
        
        Args:
            image_path: Caminho para a imagem
            species: Espécie identificada
            confidence: Confiança da identificação
            analysis_data: Dados da análise
            notes: Notas adicionais
        """
        if not os.path.exists(image_path):
            logger.error(f"Imagem não encontrada: {image_path}")
            return
        
        image_hash = self._calculate_image_hash(image_path)
        if not image_hash:
            logger.error(f"Erro ao calcular hash da imagem: {image_path}")
            return
        
        # Informações do reconhecimento
        recognition_info = {
            "image_path": image_path,
            "species": species,
            "confidence": confidence,
            "analysis_data": analysis_data,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "recognition_type": "manual_approval"
        }
        
        # Adicionar ao cache
        self.cache_data["images"][image_hash] = recognition_info
        
        # Atualizar banco de dados de espécies
        if species not in self.cache_data["species_database"]:
            self.cache_data["species_database"][species] = {
                "count": 0,
                "total_confidence": 0.0,
                "first_seen": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat()
            }
        
        species_info = self.cache_data["species_database"][species]
        species_info["count"] += 1
        species_info["total_confidence"] += confidence
        species_info["last_seen"] = datetime.now().isoformat()
        
        # Salvar cache
        self._save_cache()
        
        logger.info(f"[SUCESSO] Imagem adicionada ao cache: {os.path.basename(image_path)} -> {species}")
    
    def get_species_statistics(self) -> Dict:
        """Retorna estatísticas das espécies reconhecidas"""
        return self.cache_data["species_database"]
    
    def get_recognition_history(self, limit: int = 10) -> List[Dict]:
        """Retorna histórico de reconhecimentos"""
        recognitions = list(self.cache_data["images"].values())
        recognitions.sort(key=lambda x: x["timestamp"], reverse=True)
        return recognitions[:limit]
    
    def clear_cache(self):
        """Limpa o cache"""
        self.cache_data = {"images": {}, "species_database": {}}
        self._save_cache()
        logger.info("🗑️ Cache limpo")
    
    def export_cache(self, export_path: str):
        """Exporta cache para arquivo"""
        try:
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(self.cache_data, f, indent=2, ensure_ascii=False)
            logger.info(f"[UPLOAD] Cache exportado para: {export_path}")
        except Exception as e:
            logger.error(f"Erro ao exportar cache: {e}")

# Instância global do cache
image_cache = ImageRecognitionCache()
