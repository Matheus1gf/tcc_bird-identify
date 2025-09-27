#!/usr/bin/env python3
"""
Sistema de Memória Episódica
============================

Este módulo implementa um sistema avançado de memória episódica que permite:
- Armazenamento de experiências específicas e eventos únicos
- Recuperação de memórias baseada em contexto e similaridade
- Raciocínio sobre experiências passadas
- Sistema de consolidação de memórias
- Sistema de esquecimento adaptativo
- Memórias associativas e contextuais
- Sistema de validação de memórias
- Integração com outros sistemas cognitivos
"""

import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, Counter
import random
import time
import re
import hashlib

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Tipos de memória episódica."""
    VISUAL = "visual"                     # Memória visual
    AUDITORY = "auditory"                 # Memória auditiva
    BEHAVIORAL = "behavioral"             # Memória comportamental
    EMOTIONAL = "emotional"               # Memória emocional
    CONTEXTUAL = "contextual"             # Memória contextual
    ASSOCIATIVE = "associative"           # Memória associativa
    TEMPORAL = "temporal"                 # Memória temporal
    SPATIAL = "spatial"                   # Memória espacial

class MemoryStrength(Enum):
    """Força da memória."""
    WEAK = "weak"                         # Fraca (0.0 - 0.3)
    MODERATE = "moderate"                 # Moderada (0.3 - 0.6)
    STRONG = "strong"                     # Forte (0.6 - 0.8)
    VERY_STRONG = "very_strong"           # Muito forte (0.8 - 1.0)

class RetrievalType(Enum):
    """Tipos de recuperação de memória."""
    FREE_RECALL = "free_recall"           # Lembrança livre
    CUED_RECALL = "cued_recall"           # Lembrança com pistas
    RECOGNITION = "recognition"            # Reconhecimento
    ASSOCIATIVE = "associative"           # Associativa
    CONTEXTUAL = "contextual"             # Contextual

@dataclass
class EpisodicMemory:
    """Representa uma memória episódica."""
    memory_id: str
    memory_type: MemoryType
    content: Dict[str, Any]
    context: Dict[str, Any]
    timestamp: float
    location: Optional[str]
    emotional_state: Optional[str]
    associated_memories: List[str]
    memory_strength: MemoryStrength
    confidence: float
    access_count: int
    last_accessed: float
    consolidation_level: float
    forgetting_rate: float
    created_at: float
    last_updated: float

@dataclass
class MemoryRetrieval:
    """Resultado de recuperação de memória."""
    retrieval_id: str
    query: Dict[str, Any]
    retrieved_memories: List[str]
    retrieval_type: RetrievalType
    confidence: float
    context_similarity: float
    temporal_relevance: float
    timestamp: float

class EpisodicMemorySystem:
    """Sistema de memória episódica."""
    
    def __init__(self):
        self.memories: Dict[str, EpisodicMemory] = {}
        self.memory_index: Dict[str, Set[str]] = defaultdict(set)
        self.retrieval_history: List[MemoryRetrieval] = []
        self.consolidation_queue: List[str] = []
        self.forgetting_schedule: Dict[str, float] = {}
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar memórias básicas
        self._initialize_basic_memories()
    
    def store_episodic_memory(self, memory_type: str, content: Dict[str, Any], 
                            context: Dict[str, Any] = None, location: str = None,
                            emotional_state: str = None) -> Dict[str, Any]:
        """Armazena uma memória episódica."""
        try:
            # Converter string para enum
            memory_type_enum = MemoryType(memory_type)
            
            # Gerar ID único para a memória
            memory_id = self._generate_memory_id(content, context)
            
            # Calcular força inicial da memória
            initial_strength = self._calculate_initial_strength(content, context, emotional_state)
            strength_level = self._determine_strength_level(initial_strength)
            
            # Calcular confiança inicial
            confidence = self._calculate_initial_confidence(content, context)
            
            # Calcular taxa de esquecimento
            forgetting_rate = self._calculate_forgetting_rate(memory_type_enum, emotional_state)
            
            # Criar memória episódica
            memory = EpisodicMemory(
                memory_id=memory_id,
                memory_type=memory_type_enum,
                content=content,
                context=context or {},
                timestamp=time.time(),
                location=location,
                emotional_state=emotional_state,
                associated_memories=[],
                memory_strength=strength_level,
                confidence=confidence,
                access_count=0,
                last_accessed=time.time(),
                consolidation_level=0.0,
                forgetting_rate=forgetting_rate,
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # Armazenar memória
            self.memories[memory_id] = memory
            
            # Atualizar índices
            self._update_memory_indexes(memory)
            
            # Adicionar à fila de consolidação
            self.consolidation_queue.append(memory_id)
            
            # Agendar esquecimento
            self._schedule_forgetting(memory_id, forgetting_rate)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "memory_id": memory_id,
                "memory_type": memory_type_enum.value,
                "strength": initial_strength,
                "strength_level": strength_level.value,
                "confidence": confidence,
                "forgetting_rate": forgetting_rate,
                "message": f"Memória episódica '{memory_type}' armazenada com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro no armazenamento de memória episódica: {e}")
            return {"error": str(e)}
    
    def _generate_memory_id(self, content: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Gera ID único para a memória."""
        try:
            # Combinar conteúdo e contexto para gerar hash
            combined_data = {
                "content": content,
                "context": context,
                "timestamp": time.time()
            }
            
            # Gerar hash MD5
            data_string = json.dumps(combined_data, sort_keys=True)
            hash_object = hashlib.md5(data_string.encode())
            memory_id = f"episodic_{hash_object.hexdigest()[:12]}"
            
            return memory_id
            
        except Exception as e:
            logger.error(f"Erro na geração de ID de memória: {e}")
            return f"episodic_{int(time.time())}_{random.randint(1000, 9999)}"
    
    def _calculate_initial_strength(self, content: Dict[str, Any], 
                                  context: Dict[str, Any], 
                                  emotional_state: str = None) -> float:
        """Calcula força inicial da memória."""
        try:
            # Fator baseado no conteúdo
            content_factor = min(len(str(content)) / 1000.0, 1.0)
            
            # Fator baseado no contexto
            context_factor = min(len(str(context)) / 500.0, 1.0)
            
            # Fator emocional
            emotional_factor = 0.5
            if emotional_state:
                emotional_weights = {
                    "happy": 0.9,
                    "excited": 0.8,
                    "surprised": 0.7,
                    "neutral": 0.5,
                    "sad": 0.6,
                    "angry": 0.7,
                    "fearful": 0.8
                }
                emotional_factor = emotional_weights.get(emotional_state, 0.5)
            
            # Calcular força inicial
            initial_strength = (content_factor * 0.4 + 
                             context_factor * 0.3 + 
                             emotional_factor * 0.3)
            
            return min(initial_strength, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de força inicial: {e}")
            return 0.5
    
    def _determine_strength_level(self, strength: float) -> MemoryStrength:
        """Determina nível de força da memória."""
        if strength >= 0.8:
            return MemoryStrength.VERY_STRONG
        elif strength >= 0.6:
            return MemoryStrength.STRONG
        elif strength >= 0.3:
            return MemoryStrength.MODERATE
        else:
            return MemoryStrength.WEAK
    
    def _calculate_initial_confidence(self, content: Dict[str, Any], 
                                    context: Dict[str, Any]) -> float:
        """Calcula confiança inicial da memória."""
        try:
            # Baseado na completude dos dados
            content_completeness = len(content) / 10.0  # Normalizar
            context_completeness = len(context) / 5.0   # Normalizar
            
            # Calcular confiança
            confidence = (content_completeness * 0.6 + context_completeness * 0.4)
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança inicial: {e}")
            return 0.5
    
    def _calculate_forgetting_rate(self, memory_type: MemoryType, 
                                emotional_state: str = None) -> float:
        """Calcula taxa de esquecimento."""
        try:
            # Taxa base por tipo de memória
            base_rates = {
                MemoryType.VISUAL: 0.1,
                MemoryType.AUDITORY: 0.15,
                MemoryType.BEHAVIORAL: 0.2,
                MemoryType.EMOTIONAL: 0.05,
                MemoryType.CONTEXTUAL: 0.12,
                MemoryType.ASSOCIATIVE: 0.08,
                MemoryType.TEMPORAL: 0.18,
                MemoryType.SPATIAL: 0.1
            }
            
            base_rate = base_rates.get(memory_type, 0.1)
            
            # Ajustar por estado emocional
            if emotional_state:
                emotional_modifiers = {
                    "happy": 0.5,      # Memórias felizes são mais persistentes
                    "excited": 0.6,
                    "surprised": 0.7,
                    "neutral": 1.0,
                    "sad": 0.8,
                    "angry": 0.9,
                    "fearful": 0.3     # Memórias de medo são muito persistentes
                }
                modifier = emotional_modifiers.get(emotional_state, 1.0)
                base_rate *= modifier
            
            return min(base_rate, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de taxa de esquecimento: {e}")
            return 0.1
    
    def _update_memory_indexes(self, memory: EpisodicMemory):
        """Atualiza índices de memória."""
        try:
            # Índice por tipo
            self.memory_index[f"type_{memory.memory_type.value}"].add(memory.memory_id)
            
            # Índice por localização
            if memory.location:
                self.memory_index[f"location_{memory.location}"].add(memory.memory_id)
            
            # Índice por estado emocional
            if memory.emotional_state:
                self.memory_index[f"emotion_{memory.emotional_state}"].add(memory.memory_id)
            
            # Índice por palavras-chave do conteúdo
            content_keywords = self._extract_keywords(memory.content)
            for keyword in content_keywords:
                self.memory_index[f"keyword_{keyword}"].add(memory.memory_id)
            
            # Índice por palavras-chave do contexto
            context_keywords = self._extract_keywords(memory.context)
            for keyword in context_keywords:
                self.memory_index[f"context_{keyword}"].add(memory.memory_id)
            
        except Exception as e:
            logger.error(f"Erro na atualização de índices: {e}")
    
    def _extract_keywords(self, data: Any) -> List[str]:
        """Extrai palavras-chave dos dados."""
        keywords = []
        
        try:
            if isinstance(data, dict):
                for key, value in data.items():
                    if isinstance(value, str):
                        # Dividir string em palavras
                        words = value.lower().split()
                        keywords.extend(words[:3])  # Primeiras 3 palavras
                    elif isinstance(value, (int, float)):
                        keywords.append(str(value))
                    elif isinstance(value, list):
                        for item in value[:3]:  # Primeiros 3 itens
                            if isinstance(item, str):
                                keywords.append(item.lower())
            elif isinstance(data, list):
                for item in data[:5]:  # Primeiros 5 itens
                    if isinstance(item, str):
                        keywords.append(item.lower())
            elif isinstance(data, str):
                words = data.lower().split()
                keywords.extend(words[:5])  # Primeiras 5 palavras
            
            # Remover duplicatas e limitar
            return list(set(keywords))[:10]
            
        except Exception as e:
            logger.error(f"Erro na extração de palavras-chave: {e}")
            return []
    
    def _schedule_forgetting(self, memory_id: str, forgetting_rate: float):
        """Agenda esquecimento da memória."""
        try:
            # Calcular tempo até esquecimento (baseado na taxa)
            forgetting_time = time.time() + (1.0 / forgetting_rate) * 3600  # Horas
            self.forgetting_schedule[memory_id] = forgetting_time
            
        except Exception as e:
            logger.error(f"Erro no agendamento de esquecimento: {e}")
    
    def retrieve_memories(self, query: Dict[str, Any], 
                         retrieval_type: str = "free_recall",
                         limit: int = 10) -> Dict[str, Any]:
        """Recupera memórias baseado em uma consulta."""
        try:
            # Converter string para enum
            retrieval_type_enum = RetrievalType(retrieval_type)
            
            # Encontrar memórias relevantes
            relevant_memories = self._find_relevant_memories(query, retrieval_type_enum)
            
            # Ordenar por relevância
            scored_memories = self._score_memories(relevant_memories, query)
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            # Limitar resultados
            top_memories = scored_memories[:limit]
            
            # Atualizar contadores de acesso
            for memory_id, score in top_memories:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    memory.access_count += 1
                    memory.last_accessed = time.time()
            
            # Criar resultado de recuperação
            retrieval_result = MemoryRetrieval(
                retrieval_id=f"retrieval_{int(time.time())}_{random.randint(1000, 9999)}",
                query=query,
                retrieved_memories=[memory_id for memory_id, _ in top_memories],
                retrieval_type=retrieval_type_enum,
                confidence=self._calculate_retrieval_confidence(top_memories),
                context_similarity=self._calculate_context_similarity(query, top_memories),
                temporal_relevance=self._calculate_temporal_relevance(top_memories),
                timestamp=time.time()
            )
            
            self.retrieval_history.append(retrieval_result)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "retrieval_id": retrieval_result.retrieval_id,
                "retrieval_type": retrieval_type_enum.value,
                "retrieved_memories": len(top_memories),
                "memories": [
                    {
                        "memory_id": memory_id,
                        "memory_type": self.memories[memory_id].memory_type.value,
                        "content": self.memories[memory_id].content,
                        "context": self.memories[memory_id].context,
                        "timestamp": self.memories[memory_id].timestamp,
                        "confidence": self.memories[memory_id].confidence,
                        "strength": self.memories[memory_id].memory_strength.value,
                        "relevance_score": score
                    }
                    for memory_id, score in top_memories
                ],
                "retrieval_confidence": retrieval_result.confidence,
                "context_similarity": retrieval_result.context_similarity,
                "temporal_relevance": retrieval_result.temporal_relevance
            }
            
        except Exception as e:
            logger.error(f"Erro na recuperação de memórias: {e}")
            return {"error": str(e)}
    
    def _find_relevant_memories(self, query: Dict[str, Any], 
                              retrieval_type: RetrievalType) -> List[str]:
        """Encontra memórias relevantes para a consulta."""
        relevant_memories = set()
        
        try:
            # Busca por tipo de memória
            if "memory_type" in query:
                memory_type = query["memory_type"]
                relevant_memories.update(self.memory_index.get(f"type_{memory_type}", set()))
            
            # Busca por localização
            if "location" in query:
                location = query["location"]
                relevant_memories.update(self.memory_index.get(f"location_{location}", set()))
            
            # Busca por estado emocional
            if "emotional_state" in query:
                emotion = query["emotional_state"]
                relevant_memories.update(self.memory_index.get(f"emotion_{emotion}", set()))
            
            # Busca por palavras-chave
            if "keywords" in query:
                keywords = query["keywords"]
                for keyword in keywords:
                    relevant_memories.update(self.memory_index.get(f"keyword_{keyword}", set()))
                    relevant_memories.update(self.memory_index.get(f"context_{keyword}", set()))
            
            # Busca temporal
            if "time_range" in query:
                time_range = query["time_range"]
                current_time = time.time()
                for memory_id, memory in self.memories.items():
                    if time_range[0] <= memory.timestamp <= time_range[1]:
                        relevant_memories.add(memory_id)
            
            # Se não encontrou memórias específicas, usar todas
            if not relevant_memories:
                relevant_memories = set(self.memories.keys())
            
            return list(relevant_memories)
            
        except Exception as e:
            logger.error(f"Erro na busca de memórias relevantes: {e}")
            return list(self.memories.keys())
    
    def _score_memories(self, memory_ids: List[str], query: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Calcula scores de relevância para as memórias."""
        scored_memories = []
        
        try:
            for memory_id in memory_ids:
                if memory_id not in self.memories:
                    continue
                
                memory = self.memories[memory_id]
                
                # Calcular score baseado em múltiplos fatores
                content_similarity = self._calculate_content_similarity(memory.content, query)
                context_similarity = self._calculate_context_similarity(memory.context, query)
                temporal_relevance = self._calculate_temporal_relevance_single(memory)
                strength_factor = self._get_strength_factor(memory.memory_strength)
                access_factor = min(memory.access_count / 10.0, 1.0)
                
                # Score combinado
                score = (content_similarity * 0.3 + 
                        context_similarity * 0.25 + 
                        temporal_relevance * 0.2 + 
                        strength_factor * 0.15 + 
                        access_factor * 0.1)
                
                scored_memories.append((memory_id, score))
            
            return scored_memories
            
        except Exception as e:
            logger.error(f"Erro no cálculo de scores: {e}")
            return []
    
    def _calculate_content_similarity(self, content: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Calcula similaridade de conteúdo."""
        try:
            if not content or not query:
                return 0.0
            
            # Extrair palavras-chave
            content_keywords = self._extract_keywords(content)
            query_keywords = self._extract_keywords(query)
            
            if not content_keywords or not query_keywords:
                return 0.0
            
            # Calcular similaridade de Jaccard
            content_set = set(content_keywords)
            query_set = set(query_keywords)
            
            intersection = len(content_set & query_set)
            union = len(content_set | query_set)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Erro no cálculo de similaridade de conteúdo: {e}")
            return 0.0
    
    def _calculate_context_similarity(self, context: Dict[str, Any], query: Dict[str, Any]) -> float:
        """Calcula similaridade de contexto."""
        return self._calculate_content_similarity(context, query)
    
    def _calculate_temporal_relevance_single(self, memory: EpisodicMemory) -> float:
        """Calcula relevância temporal de uma memória."""
        try:
            current_time = time.time()
            time_diff = current_time - memory.timestamp
            
            # Memórias mais recentes têm maior relevância
            # Decaimento exponencial
            relevance = np.exp(-time_diff / (30 * 24 * 3600))  # 30 dias
            
            return min(relevance, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de relevância temporal: {e}")
            return 0.0
    
    def _calculate_temporal_relevance(self, scored_memories: List[Tuple[str, float]]) -> float:
        """Calcula relevância temporal geral."""
        try:
            if not scored_memories:
                return 0.0
            
            total_relevance = 0.0
            for memory_id, score in scored_memories:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    temporal_relevance = self._calculate_temporal_relevance_single(memory)
                    total_relevance += temporal_relevance * score
            
            return total_relevance / len(scored_memories)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de relevância temporal geral: {e}")
            return 0.0
    
    def _get_strength_factor(self, strength: MemoryStrength) -> float:
        """Obtém fator de força."""
        strength_factors = {
            MemoryStrength.WEAK: 0.2,
            MemoryStrength.MODERATE: 0.5,
            MemoryStrength.STRONG: 0.8,
            MemoryStrength.VERY_STRONG: 1.0
        }
        return strength_factors.get(strength, 0.5)
    
    def _calculate_retrieval_confidence(self, scored_memories: List[Tuple[str, float]]) -> float:
        """Calcula confiança na recuperação."""
        try:
            if not scored_memories:
                return 0.0
            
            # Confiança baseada na qualidade dos resultados
            total_score = sum(score for _, score in scored_memories)
            average_score = total_score / len(scored_memories)
            
            # Confiança baseada na consistência
            scores = [score for _, score in scored_memories]
            score_variance = np.var(scores) if len(scores) > 1 else 0.0
            consistency_factor = 1.0 - min(score_variance, 1.0)
            
            confidence = average_score * consistency_factor
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança de recuperação: {e}")
            return 0.0
    
    def consolidate_memories(self) -> Dict[str, Any]:
        """Consolida memórias na fila de consolidação."""
        try:
            consolidated_count = 0
            
            for memory_id in self.consolidation_queue[:]:
                if memory_id in self.memories:
                    memory = self.memories[memory_id]
                    
                    # Aumentar nível de consolidação
                    memory.consolidation_level = min(memory.consolidation_level + 0.1, 1.0)
                    
                    # Aumentar força se consolidada
                    if memory.consolidation_level >= 0.8:
                        current_strength = self._get_strength_factor(memory.memory_strength)
                        if current_strength < 0.8:
                            memory.memory_strength = MemoryStrength.STRONG
                        elif current_strength < 1.0:
                            memory.memory_strength = MemoryStrength.VERY_STRONG
                    
                    memory.last_updated = time.time()
                    consolidated_count += 1
                    
                    # Remover da fila se totalmente consolidada
                    if memory.consolidation_level >= 1.0:
                        self.consolidation_queue.remove(memory_id)
            
            return {
                "success": True,
                "consolidated_memories": consolidated_count,
                "remaining_in_queue": len(self.consolidation_queue),
                "message": f"{consolidated_count} memórias consolidadas"
            }
            
        except Exception as e:
            logger.error(f"Erro na consolidação de memórias: {e}")
            return {"error": str(e)}
    
    def forget_old_memories(self) -> Dict[str, Any]:
        """Esquece memórias antigas baseado no agendamento."""
        try:
            current_time = time.time()
            forgotten_memories = []
            
            for memory_id, forget_time in list(self.forgetting_schedule.items()):
                if current_time >= forget_time:
                    if memory_id in self.memories:
                        # Verificar se memória ainda é relevante
                        memory = self.memories[memory_id]
                        
                        # Não esquecer memórias muito acessadas ou muito fortes
                        if memory.access_count < 5 and memory.memory_strength in [MemoryStrength.WEAK, MemoryStrength.MODERATE]:
                            del self.memories[memory_id]
                            forgotten_memories.append(memory_id)
                            
                            # Remover dos índices
                            self._remove_from_indexes(memory_id)
                            
                            # Remover do agendamento
                            del self.forgetting_schedule[memory_id]
            
            return {
                "success": True,
                "forgotten_memories": len(forgotten_memories),
                "remaining_memories": len(self.memories),
                "message": f"{len(forgotten_memories)} memórias esquecidas"
            }
            
        except Exception as e:
            logger.error(f"Erro no esquecimento de memórias: {e}")
            return {"error": str(e)}
    
    def _remove_from_indexes(self, memory_id: str):
        """Remove memória dos índices."""
        try:
            for index_name, memory_set in self.memory_index.items():
                memory_set.discard(memory_id)
        except Exception as e:
            logger.error(f"Erro na remoção de índices: {e}")
    
    def get_memory_analysis(self, memory_id: str = None) -> Dict[str, Any]:
        """Obtém análise de memórias."""
        try:
            if memory_id:
                # Análise específica de uma memória
                if memory_id not in self.memories:
                    return {"error": f"Memória '{memory_id}' não encontrada"}
                
                memory = self.memories[memory_id]
                
                return {
                    "memory": asdict(memory),
                    "age_days": (time.time() - memory.timestamp) / (24 * 3600),
                    "access_frequency": memory.access_count / max(1, (time.time() - memory.created_at) / (24 * 3600)),
                    "consolidation_progress": memory.consolidation_level * 100,
                    "forgetting_schedule": self.forgetting_schedule.get(memory_id, None)
                }
            else:
                # Análise geral
                return {
                    "total_memories": len(self.memories),
                    "total_retrievals": len(self.retrieval_history),
                    "consolidation_queue_size": len(self.consolidation_queue),
                    "memory_distribution": self._get_memory_distribution(),
                    "strength_distribution": self._get_strength_distribution(),
                    "type_distribution": self._get_type_distribution(),
                    "memory_statistics": self._get_memory_statistics()
                }
                
        except Exception as e:
            logger.error(f"Erro na análise de memórias: {e}")
            return {"error": str(e)}
    
    def _get_memory_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de memórias por tipo."""
        distribution = defaultdict(int)
        
        for memory in self.memories.values():
            distribution[memory.memory_type.value] += 1
        
        return dict(distribution)
    
    def _get_strength_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de memórias por força."""
        distribution = defaultdict(int)
        
        for memory in self.memories.values():
            distribution[memory.memory_strength.value] += 1
        
        return dict(distribution)
    
    def _get_type_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de tipos de recuperação."""
        distribution = defaultdict(int)
        
        for retrieval in self.retrieval_history:
            distribution[retrieval.retrieval_type.value] += 1
        
        return dict(distribution)
    
    def _get_memory_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas das memórias."""
        try:
            if not self.memories:
                return {"total_memories": 0, "average_age": 0, "average_access": 0}
            
            current_time = time.time()
            ages = []
            access_counts = []
            confidences = []
            
            for memory in self.memories.values():
                ages.append((current_time - memory.timestamp) / (24 * 3600))  # Dias
                access_counts.append(memory.access_count)
                confidences.append(memory.confidence)
            
            return {
                "total_memories": len(self.memories),
                "average_age_days": sum(ages) / len(ages),
                "average_access_count": sum(access_counts) / len(access_counts),
                "average_confidence": sum(confidences) / len(confidences),
                "oldest_memory_days": max(ages),
                "newest_memory_days": min(ages),
                "most_accessed": max(access_counts),
                "least_accessed": min(access_counts)
            }
            
        except Exception as e:
            logger.error(f"Erro nas estatísticas de memórias: {e}")
            return {"error": str(e)}
    
    def _initialize_basic_memories(self):
        """Inicializa memórias básicas."""
        # Memórias básicas do sistema
        basic_memories = [
            {
                "memory_type": "visual",
                "content": {"description": "Primeira detecção de pássaro", "species": "unknown"},
                "context": {"environment": "initialization", "confidence": 0.5},
                "emotional_state": "neutral"
            },
            {
                "memory_type": "behavioral",
                "content": {"action": "system_startup", "timestamp": time.time()},
                "context": {"system_state": "initialization"},
                "emotional_state": "neutral"
            }
        ]
        
        for memory_data in basic_memories:
            try:
                self.store_episodic_memory(**memory_data)
            except Exception as e:
                logger.error(f"Erro na inicialização de memória básica: {e}")
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/episodic_memory.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar memórias
                if "memories" in data:
                    for memory_id, memory_data in data["memories"].items():
                        memory_data["memory_type"] = MemoryType(memory_data["memory_type"])
                        memory_data["memory_strength"] = MemoryStrength(memory_data["memory_strength"])
                        self.memories[memory_id] = EpisodicMemory(**memory_data)
                
                # Carregar histórico de recuperação
                if "retrieval_history" in data:
                    for retrieval_data in data["retrieval_history"]:
                        retrieval_data["retrieval_type"] = RetrievalType(retrieval_data["retrieval_type"])
                        self.retrieval_history.append(MemoryRetrieval(**retrieval_data))
                
                # Carregar outros dados
                if "consolidation_queue" in data:
                    self.consolidation_queue = data["consolidation_queue"]
                
                if "forgetting_schedule" in data:
                    self.forgetting_schedule = data["forgetting_schedule"]
                
                logger.info(f"Dados de memória episódica carregados: {len(self.memories)} memórias")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de memória episódica: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/episodic_memory.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "memories": {
                    memory_id: {
                        "memory_id": memory.memory_id,
                        "memory_type": memory.memory_type.value,
                        "content": memory.content,
                        "context": memory.context,
                        "timestamp": memory.timestamp,
                        "location": memory.location,
                        "emotional_state": memory.emotional_state,
                        "associated_memories": memory.associated_memories,
                        "memory_strength": memory.memory_strength.value,
                        "confidence": memory.confidence,
                        "access_count": memory.access_count,
                        "last_accessed": memory.last_accessed,
                        "consolidation_level": memory.consolidation_level,
                        "forgetting_rate": memory.forgetting_rate,
                        "created_at": memory.created_at,
                        "last_updated": memory.last_updated
                    }
                    for memory_id, memory in self.memories.items()
                },
                "retrieval_history": [
                    {
                        "retrieval_id": retrieval.retrieval_id,
                        "query": retrieval.query,
                        "retrieved_memories": retrieval.retrieved_memories,
                        "retrieval_type": retrieval.retrieval_type.value,
                        "confidence": retrieval.confidence,
                        "context_similarity": retrieval.context_similarity,
                        "temporal_relevance": retrieval.temporal_relevance,
                        "timestamp": retrieval.timestamp
                    }
                    for retrieval in self.retrieval_history
                ],
                "consolidation_queue": self.consolidation_queue,
                "forgetting_schedule": self.forgetting_schedule,
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de memória episódica salvos: {len(self.memories)} memórias")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de memória episódica: {e}")
