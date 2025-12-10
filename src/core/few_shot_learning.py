#!/usr/bin/env python3
"""
Sistema de Aprendizado Few-Shot
==============================

Este módulo implementa um sistema avançado de aprendizado few-shot que permite:
- Aprendizado de novos conceitos com poucos exemplos
- Transferência de conhecimento de conceitos similares
- Meta-aprendizado para melhorar estratégias de few-shot
- Sistema de protótipos para representação de conceitos
- Sistema de similaridade para encontrar conceitos relacionados
- Sistema de adaptação rápida para novos domínios
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
import pickle

# Configurar logging primeiro
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Tentar importar sklearn, mas não falhar se não estiver disponível
try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn não disponível - funcionalidades de similaridade limitadas")

class FewShotStrategy(Enum):
    """Estratégias de aprendizado few-shot."""
    PROTOTYPE_BASED = "prototype_based"      # Baseado em protótipos
    SIMILARITY_BASED = "similarity_based"    # Baseado em similaridade
    META_LEARNING = "meta_learning"          # Meta-aprendizado
    TRANSFER_LEARNING = "transfer_learning"  # Transferência de aprendizado
    ADAPTIVE_FUSION = "adaptive_fusion"      # Fusão adaptativa

class LearningMode(Enum):
    """Modos de aprendizado."""
    ZERO_SHOT = "zero_shot"        # Aprendizado sem exemplos
    ONE_SHOT = "one_shot"          # Aprendizado com 1 exemplo
    FEW_SHOT = "few_shot"          # Aprendizado com poucos exemplos (2-5)
    MANY_SHOT = "many_shot"        # Aprendizado com muitos exemplos (>5)

class ConceptType(Enum):
    """Tipos de conceitos."""
    SPECIES = "species"            # Espécies de animais
    OBJECT = "object"              # Objetos
    BEHAVIOR = "behavior"          # Comportamentos
    PATTERN = "pattern"           # Padrões
    FEATURE = "feature"           # Características

@dataclass
class FewShotExample:
    """Exemplo para aprendizado few-shot."""
    concept_name: str
    concept_type: ConceptType
    detection_data: Dict[str, Any]
    features: List[float]
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class ConceptPrototype:
    """Protótipo de um conceito."""
    concept_name: str
    concept_type: ConceptType
    prototype_features: List[float]
    support_examples: List[FewShotExample]
    confidence: float
    learning_mode: LearningMode
    created_at: float
    last_updated: float
    usage_count: int

@dataclass
class FewShotResult:
    """Resultado de aprendizado few-shot."""
    concept_name: str
    success: bool
    confidence: float
    learning_mode: LearningMode
    strategy_used: FewShotStrategy
    examples_used: int
    transfer_sources: List[str]
    adaptation_required: bool
    performance_improvement: float
    timestamp: float

class FeatureExtractor:
    """Extrator de características para few-shot learning."""
    
    def __init__(self):
        self.feature_dimensions = 128  # Dimensão das características extraídas
        self.feature_cache: Dict[str, List[float]] = {}
    
    def extract_features(self, detection_data: Dict[str, Any]) -> List[float]:
        """Extrai características de dados de detecção."""
        try:
            features = []
            
            # Características de forma
            if "shape_analysis" in detection_data:
                shape_data = detection_data["shape_analysis"]
                features.extend(self._extract_shape_features(shape_data))
            
            # Características de cor
            if "color_analysis" in detection_data:
                color_data = detection_data["color_analysis"]
                features.extend(self._extract_color_features(color_data))
            
            # Características de textura
            if "texture_analysis" in detection_data:
                texture_data = detection_data["texture_analysis"]
                features.extend(self._extract_texture_features(texture_data))
            
            # Características de padrão
            if "pattern_analysis" in detection_data:
                pattern_data = detection_data["pattern_analysis"]
                features.extend(self._extract_pattern_features(pattern_data))
            
            # Características de comportamento
            if "behavior_analysis" in detection_data:
                behavior_data = detection_data["behavior_analysis"]
                features.extend(self._extract_behavior_features(behavior_data))
            
            # Características de movimento
            if "motion_analysis" in detection_data:
                motion_data = detection_data["motion_analysis"]
                features.extend(self._extract_motion_features(motion_data))
            
            # Normalizar características para ter dimensão fixa
            features = self._normalize_features(features)
            
            return features
            
        except Exception as e:
            logger.error(f"Erro ao extrair características: {e}")
            return [0.0] * self.feature_dimensions
    
    def _extract_shape_features(self, shape_data: Dict[str, Any]) -> List[float]:
        """Extrai características de forma."""
        features = []
        
        # Características básicas de forma
        detected_shapes = shape_data.get("detected_shapes", [])
        features.append(len(detected_shapes))  # Número de formas detectadas
        features.append(shape_data.get("confidence", 0.0))  # Confiança
        
        # Características específicas
        shape_keywords = ["beak", "wing", "tail", "head", "body", "limb", "compact", "symmetrical"]
        for keyword in shape_keywords:
            features.append(1.0 if keyword in detected_shapes else 0.0)
        
        # Características de simetria
        features.append(1.0 if shape_data.get("symmetrical", False) else 0.0)
        
        return features
    
    def _extract_color_features(self, color_data: Dict[str, Any]) -> List[float]:
        """Extrai características de cor."""
        features = []
        
        detected_colors = color_data.get("detected_colors", [])
        features.append(len(detected_colors))  # Número de cores detectadas
        features.append(color_data.get("confidence", 0.0))  # Confiança
        
        # Características específicas de cor
        color_keywords = ["blue", "red", "green", "yellow", "brown", "black", "white", "gray"]
        for keyword in color_keywords:
            features.append(1.0 if keyword in detected_colors else 0.0)
        
        return features
    
    def _extract_texture_features(self, texture_data: Dict[str, Any]) -> List[float]:
        """Extrai características de textura."""
        features = []
        
        detected_textures = texture_data.get("detected_textures", [])
        features.append(len(detected_textures))  # Número de texturas detectadas
        features.append(texture_data.get("confidence", 0.0))  # Confiança
        
        # Características específicas de textura
        texture_keywords = ["feathery", "furry", "scaly", "smooth", "rough", "soft", "hard"]
        for keyword in texture_keywords:
            features.append(1.0 if keyword in detected_textures else 0.0)
        
        return features
    
    def _extract_pattern_features(self, pattern_data: Dict[str, Any]) -> List[float]:
        """Extrai características de padrão."""
        features = []
        
        detected_patterns = pattern_data.get("detected_patterns", [])
        features.append(len(detected_patterns))  # Número de padrões detectados
        features.append(pattern_data.get("confidence", 0.0))  # Confiança
        
        # Características específicas de padrão
        pattern_keywords = ["striped", "spotted", "solid", "feather_pattern", "scales", "uniform"]
        for keyword in pattern_keywords:
            features.append(1.0 if keyword in detected_patterns else 0.0)
        
        return features
    
    def _extract_behavior_features(self, behavior_data: Dict[str, Any]) -> List[float]:
        """Extrai características de comportamento."""
        features = []
        
        # Características de comportamento
        behavior_keywords = ["feeding_behavior", "flying", "walking", "swimming", "resting"]
        for keyword in behavior_keywords:
            features.append(1.0 if behavior_data.get(keyword, False) else 0.0)
        
        features.append(behavior_data.get("confidence", 0.0))  # Confiança
        
        return features
    
    def _extract_motion_features(self, motion_data: Dict[str, Any]) -> List[float]:
        """Extrai características de movimento."""
        features = []
        
        features.append(1.0 if motion_data.get("movement_detected", False) else 0.0)
        features.append(motion_data.get("confidence", 0.0))  # Confiança
        
        return features
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """Normaliza características para ter dimensão fixa."""
        # Preencher ou truncar para ter dimensão fixa
        if len(features) < self.feature_dimensions:
            features.extend([0.0] * (self.feature_dimensions - len(features)))
        elif len(features) > self.feature_dimensions:
            features = features[:self.feature_dimensions]
        
        return features

class PrototypeManager:
    """Gerenciador de protótipos para few-shot learning."""
    
    def __init__(self):
        self.prototypes: Dict[str, ConceptPrototype] = {}
        self.feature_extractor = FeatureExtractor()
        self.similarity_threshold = 0.7
    
    def create_prototype(self, concept_name: str, concept_type: ConceptType, 
                        examples: List[FewShotExample]) -> ConceptPrototype:
        """Cria um protótipo baseado em exemplos."""
        try:
            if not examples:
                raise ValueError("Pelo menos um exemplo é necessário para criar um protótipo")
            
            # Extrair características de todos os exemplos
            example_features = []
            for example in examples:
                features = self.feature_extractor.extract_features(example.detection_data)
                example_features.append(features)
            
            # Calcular protótipo como média das características
            prototype_features = np.mean(example_features, axis=0).tolist()
            
            # Determinar modo de aprendizado
            learning_mode = self._determine_learning_mode(len(examples))
            
            # Calcular confiança baseada no número de exemplos e consistência
            confidence = self._calculate_prototype_confidence(examples, example_features)
            
            # Criar protótipo
            prototype = ConceptPrototype(
                concept_name=concept_name,
                concept_type=concept_type,
                prototype_features=prototype_features,
                support_examples=examples,
                confidence=confidence,
                learning_mode=learning_mode,
                created_at=time.time(),
                last_updated=time.time(),
                usage_count=0
            )
            
            # Armazenar protótipo
            self.prototypes[concept_name] = prototype
            
            return prototype
            
        except Exception as e:
            logger.error(f"Erro ao criar protótipo: {e}")
            raise
    
    def find_similar_prototypes(self, features: List[float], concept_type: ConceptType = None) -> List[Tuple[str, float]]:
        """Encontra protótipos similares baseado nas características."""
        try:
            similarities = []
            
            for name, prototype in self.prototypes.items():
                # Filtrar por tipo de conceito se especificado
                if concept_type and prototype.concept_type != concept_type:
                    continue
                
                # Calcular similaridade
                similarity = self._calculate_similarity(features, prototype.prototype_features)
                
                if similarity >= self.similarity_threshold:
                    similarities.append((name, similarity))
            
            # Ordenar por similaridade
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return similarities
            
        except Exception as e:
            logger.error(f"Erro ao encontrar protótipos similares: {e}")
            return []
    
    def update_prototype(self, concept_name: str, new_example: FewShotExample) -> bool:
        """Atualiza um protótipo com um novo exemplo."""
        try:
            if concept_name not in self.prototypes:
                return False
            
            prototype = self.prototypes[concept_name]
            
            # Adicionar novo exemplo
            prototype.support_examples.append(new_example)
            
            # Recalcular protótipo
            example_features = []
            for example in prototype.support_examples:
                features = self.feature_extractor.extract_features(example.detection_data)
                example_features.append(features)
            
            # Atualizar características do protótipo
            prototype.prototype_features = np.mean(example_features, axis=0).tolist()
            
            # Atualizar confiança
            prototype.confidence = self._calculate_prototype_confidence(
                prototype.support_examples, example_features
            )
            
            # Atualizar modo de aprendizado
            prototype.learning_mode = self._determine_learning_mode(len(prototype.support_examples))
            
            # Atualizar timestamp
            prototype.last_updated = time.time()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao atualizar protótipo: {e}")
            return False
    
    def _determine_learning_mode(self, num_examples: int) -> LearningMode:
        """Determina o modo de aprendizado baseado no número de exemplos."""
        if num_examples == 0:
            return LearningMode.ZERO_SHOT
        elif num_examples == 1:
            return LearningMode.ONE_SHOT
        elif num_examples <= 5:
            return LearningMode.FEW_SHOT
        else:
            return LearningMode.MANY_SHOT
    
    def _calculate_prototype_confidence(self, examples: List[FewShotExample], 
                                      example_features: List[List[float]]) -> float:
        """Calcula confiança do protótipo baseado na consistência dos exemplos."""
        if not examples:
            return 0.0
        
        # Confiança baseada no número de exemplos
        base_confidence = min(len(examples) / 5.0, 1.0)
        
        # Confiança baseada na consistência das características
        if len(example_features) > 1:
            # Calcular variância das características
            feature_array = np.array(example_features)
            feature_variance = np.var(feature_array, axis=0).mean()
            
            # Confiança baseada na consistência (menor variância = maior confiança)
            consistency_confidence = max(0.0, 1.0 - feature_variance)
        else:
            consistency_confidence = 0.5  # Confiança média para um exemplo
        
        # Confiança média dos exemplos
        example_confidence = np.mean([ex.confidence for ex in examples])
        
        # Confiança final é média ponderada
        final_confidence = (base_confidence * 0.3 + 
                           consistency_confidence * 0.4 + 
                           example_confidence * 0.3)
        
        return min(final_confidence, 1.0)
    
    def _calculate_similarity(self, features1: List[float], features2: List[float]) -> float:
        """Calcula similaridade entre duas características."""
        try:
            if SKLEARN_AVAILABLE:
                # Usar sklearn se disponível
                f1 = np.array(features1).reshape(1, -1)
                f2 = np.array(features2).reshape(1, -1)
                similarity = cosine_similarity(f1, f2)[0][0]
                return float(similarity)
            else:
                # Implementação manual de similaridade de cosseno
                f1 = np.array(features1)
                f2 = np.array(features2)
                
                # Calcular produto escalar
                dot_product = np.dot(f1, f2)
                
                # Calcular normas
                norm1 = np.linalg.norm(f1)
                norm2 = np.linalg.norm(f2)
                
                # Evitar divisão por zero
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                
                # Calcular similaridade de cosseno
                similarity = dot_product / (norm1 * norm2)
                return float(similarity)
            
        except Exception as e:
            logger.error(f"Erro ao calcular similaridade: {e}")
            return 0.0

class FewShotLearner:
    """Sistema principal de aprendizado few-shot."""
    
    def __init__(self):
        self.prototype_manager = PrototypeManager()
        self.feature_extractor = FeatureExtractor()
        self.learning_history: List[FewShotResult] = []
        self.concept_knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.strategy_performance: Dict[FewShotStrategy, List[float]] = {
            strategy: [] for strategy in FewShotStrategy
        }
        
        # Carregar dados existentes
        self._load_data()
    
    def learn_concept_few_shot(self, concept_name: str, concept_type: ConceptType,
                              examples: List[Dict[str, Any]], strategy: FewShotStrategy = FewShotStrategy.PROTOTYPE_BASED) -> FewShotResult:
        """Aprende um conceito usando few-shot learning."""
        try:
            # Converter exemplos para FewShotExample
            few_shot_examples = []
            for i, example_data in enumerate(examples):
                features = self.feature_extractor.extract_features(example_data)
                confidence = self._calculate_example_confidence(example_data)
                
                few_shot_example = FewShotExample(
                    concept_name=concept_name,
                    concept_type=concept_type,
                    detection_data=example_data,
                    features=features,
                    confidence=confidence,
                    timestamp=time.time(),
                    metadata={"example_index": i}
                )
                few_shot_examples.append(few_shot_example)
            
            # Aplicar estratégia de aprendizado
            if strategy == FewShotStrategy.PROTOTYPE_BASED:
                success, confidence, transfer_sources = self._prototype_based_learning(
                    concept_name, concept_type, few_shot_examples
                )
            elif strategy == FewShotStrategy.SIMILARITY_BASED:
                success, confidence, transfer_sources = self._similarity_based_learning(
                    concept_name, concept_type, few_shot_examples
                )
            elif strategy == FewShotStrategy.META_LEARNING:
                success, confidence, transfer_sources = self._meta_learning(
                    concept_name, concept_type, few_shot_examples
                )
            elif strategy == FewShotStrategy.TRANSFER_LEARNING:
                success, confidence, transfer_sources = self._transfer_learning(
                    concept_name, concept_type, few_shot_examples
                )
            elif strategy == FewShotStrategy.ADAPTIVE_FUSION:
                success, confidence, transfer_sources = self._adaptive_fusion_learning(
                    concept_name, concept_type, few_shot_examples
                )
            else:
                success, confidence, transfer_sources = False, 0.0, []
            
            # Determinar modo de aprendizado
            learning_mode = self.prototype_manager._determine_learning_mode(len(few_shot_examples))
            
            # Calcular melhoria de performance
            performance_improvement = self._calculate_performance_improvement(success, confidence)
            
            # Criar resultado
            result = FewShotResult(
                concept_name=concept_name,
                success=success,
                confidence=confidence,
                learning_mode=learning_mode,
                strategy_used=strategy,
                examples_used=len(few_shot_examples),
                transfer_sources=transfer_sources,
                adaptation_required=strategy != FewShotStrategy.PROTOTYPE_BASED,
                performance_improvement=performance_improvement,
                timestamp=time.time()
            )
            
            # Atualizar histórico
            self.learning_history.append(result)
            
            # Atualizar performance da estratégia
            self.strategy_performance[strategy].append(confidence if success else 0.0)
            
            # Atualizar base de conhecimento
            self.concept_knowledge_base[concept_name] = {
                "concept_type": concept_type.value,
                "examples_count": len(few_shot_examples),
                "learning_mode": learning_mode.value,
                "strategy_used": strategy.value,
                "confidence": confidence,
                "success": success,
                "last_learned": time.time()
            }
            
            # Salvar dados
            self._save_data()
            
            return result
            
        except Exception as e:
            logger.error(f"Erro no aprendizado few-shot: {e}")
            return FewShotResult(
                concept_name=concept_name,
                success=False,
                confidence=0.0,
                learning_mode=LearningMode.ZERO_SHOT,
                strategy_used=strategy,
                examples_used=0,
                transfer_sources=[],
                adaptation_required=False,
                performance_improvement=0.0,
                timestamp=time.time()
            )
    
    def classify_with_few_shot(self, detection_data: Dict[str, Any], 
                               concept_type: ConceptType = None) -> Dict[str, Any]:
        """Classifica dados usando protótipos few-shot."""
        try:
            # Extrair características
            features = self.feature_extractor.extract_features(detection_data)
            
            # Encontrar protótipos similares
            similar_prototypes = self.prototype_manager.find_similar_prototypes(features, concept_type)
            
            if not similar_prototypes:
                return {
                    "classification": "unknown",
                    "confidence": 0.0,
                    "similar_concepts": [],
                    "learning_suggestion": "Nenhum conceito similar encontrado"
                }
            
            # Pegar o protótipo mais similar
            best_match = similar_prototypes[0]
            concept_name, similarity = best_match
            
            # Calcular confiança baseada na similaridade e confiança do protótipo
            prototype = self.prototype_manager.prototypes[concept_name]
            confidence = similarity * prototype.confidence
            
            # Incrementar contador de uso
            prototype.usage_count += 1
            
            return {
                "classification": concept_name,
                "confidence": confidence,
                "similarity": similarity,
                "prototype_confidence": prototype.confidence,
                "learning_mode": prototype.learning_mode.value,
                "similar_concepts": similar_prototypes[:3],  # Top 3
                "learning_suggestion": f"Conceito '{concept_name}' identificado com confiança {confidence:.3f}"
            }
            
        except Exception as e:
            logger.error(f"Erro na classificação few-shot: {e}")
            return {
                "classification": "error",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def get_few_shot_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """Obtém análise do sistema few-shot."""
        try:
            if concept_name:
                # Análise específica de um conceito
                if concept_name in self.concept_knowledge_base:
                    return self.concept_knowledge_base[concept_name]
                else:
                    return {"error": f"Conceito {concept_name} não encontrado"}
            
            # Análise geral
            total_concepts = len(self.concept_knowledge_base)
            total_prototypes = len(self.prototype_manager.prototypes)
            total_learning_sessions = len(self.learning_history)
            
            # Calcular métricas de performance
            successful_learnings = sum(1 for result in self.learning_history if result.success)
            success_rate = successful_learnings / total_learning_sessions if total_learning_sessions > 0 else 0.0
            
            average_confidence = np.mean([
                result.confidence for result in self.learning_history
            ]) if self.learning_history else 0.0
            
            # Performance por estratégia
            strategy_stats = {}
            for strategy, performances in self.strategy_performance.items():
                if performances:
                    strategy_stats[strategy.value] = {
                        "average_performance": np.mean(performances),
                        "usage_count": len(performances),
                        "best_performance": max(performances)
                    }
                else:
                    strategy_stats[strategy.value] = {
                        "average_performance": 0.0,
                        "usage_count": 0,
                        "best_performance": 0.0
                    }
            
            # Conceitos por modo de aprendizado
            learning_mode_distribution = Counter([
                result.learning_mode.value for result in self.learning_history
            ])
            
            return {
                "total_concepts": total_concepts,
                "total_prototypes": total_prototypes,
                "total_learning_sessions": total_learning_sessions,
                "success_rate": success_rate,
                "average_confidence": average_confidence,
                "strategy_performance": strategy_stats,
                "learning_mode_distribution": dict(learning_mode_distribution),
                "recent_learnings": [{
                    "concept_name": result.concept_name,
                    "success": result.success,
                    "confidence": result.confidence,
                    "learning_mode": result.learning_mode.value,
                    "strategy_used": result.strategy_used.value,
                    "examples_used": result.examples_used,
                    "transfer_sources": result.transfer_sources,
                    "adaptation_required": result.adaptation_required,
                    "performance_improvement": result.performance_improvement,
                    "timestamp": result.timestamp
                } for result in self.learning_history[-5:]],
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise few-shot: {e}")
            return {"error": str(e)}
    
    def _prototype_based_learning(self, concept_name: str, concept_type: ConceptType, 
                                 examples: List[FewShotExample]) -> Tuple[bool, float, List[str]]:
        """Aprendizado baseado em protótipos."""
        try:
            # Criar protótipo
            prototype = self.prototype_manager.create_prototype(concept_name, concept_type, examples)
            
            # Calcular confiança baseada na qualidade do protótipo
            confidence = prototype.confidence
            
            # Verificar se há protótipos similares para transferência
            similar_prototypes = self.prototype_manager.find_similar_prototypes(
                prototype.prototype_features, concept_type
            )
            
            transfer_sources = [name for name, _ in similar_prototypes[:3]]
            
            return True, confidence, transfer_sources
            
        except Exception as e:
            logger.error(f"Erro no aprendizado baseado em protótipos: {e}")
            return False, 0.0, []
    
    def _similarity_based_learning(self, concept_name: str, concept_type: ConceptType, 
                                  examples: List[FewShotExample]) -> Tuple[bool, float, List[str]]:
        """Aprendizado baseado em similaridade."""
        try:
            if not examples:
                return False, 0.0, []
            
            # Usar o primeiro exemplo como referência
            reference_example = examples[0]
            
            # Encontrar conceitos similares
            similar_prototypes = self.prototype_manager.find_similar_prototypes(
                reference_example.features, concept_type
            )
            
            if not similar_prototypes:
                # Criar novo protótipo se não há similares
                prototype = self.prototype_manager.create_prototype(concept_name, concept_type, examples)
                return True, prototype.confidence, []
            
            # Usar similaridade para adaptar conceito existente
            best_similarity = similar_prototypes[0][1]
            confidence = best_similarity * 0.8  # Confiança reduzida por adaptação
            
            transfer_sources = [name for name, _ in similar_prototypes[:3]]
            
            return True, confidence, transfer_sources
            
        except Exception as e:
            logger.error(f"Erro no aprendizado baseado em similaridade: {e}")
            return False, 0.0, []
    
    def _meta_learning(self, concept_name: str, concept_type: ConceptType, 
                       examples: List[FewShotExample]) -> Tuple[bool, float, List[str]]:
        """Meta-aprendizado para melhorar estratégias."""
        try:
            # Analisar histórico de aprendizado para escolher melhor estratégia
            best_strategy = self._get_best_strategy_for_concept_type(concept_type)
            
            # Aplicar estratégia escolhida
            if best_strategy == FewShotStrategy.PROTOTYPE_BASED:
                return self._prototype_based_learning(concept_name, concept_type, examples)
            elif best_strategy == FewShotStrategy.SIMILARITY_BASED:
                return self._similarity_based_learning(concept_name, concept_type, examples)
            else:
                # Fallback para protótipo
                return self._prototype_based_learning(concept_name, concept_type, examples)
            
        except Exception as e:
            logger.error(f"Erro no meta-aprendizado: {e}")
            return False, 0.0, []
    
    def _transfer_learning(self, concept_name: str, concept_type: ConceptType, 
                          examples: List[FewShotExample]) -> Tuple[bool, float, List[str]]:
        """Transferência de aprendizado de conceitos similares."""
        try:
            if not examples:
                return False, 0.0, []
            
            # Encontrar conceitos similares para transferência
            reference_example = examples[0]
            similar_prototypes = self.prototype_manager.find_similar_prototypes(
                reference_example.features, concept_type
            )
            
            if not similar_prototypes:
                # Criar novo protótipo se não há similares
                prototype = self.prototype_manager.create_prototype(concept_name, concept_type, examples)
                return True, prototype.confidence, []
            
            # Transferir conhecimento do conceito mais similar
            best_similar = similar_prototypes[0]
            source_concept = best_similar[0]
            similarity = best_similar[1]
            
            # Criar protótipo adaptado
            prototype = self.prototype_manager.create_prototype(concept_name, concept_type, examples)
            
            # Ajustar confiança baseada na transferência
            transfer_confidence = prototype.confidence * similarity * 0.9
            
            transfer_sources = [source_concept]
            
            return True, transfer_confidence, transfer_sources
            
        except Exception as e:
            logger.error(f"Erro na transferência de aprendizado: {e}")
            return False, 0.0, []
    
    def _adaptive_fusion_learning(self, concept_name: str, concept_type: ConceptType, 
                                 examples: List[FewShotExample]) -> Tuple[bool, float, List[str]]:
        """Aprendizado por fusão adaptativa de múltiplas estratégias."""
        try:
            # Aplicar múltiplas estratégias
            results = []
            
            # Protótipo
            success1, conf1, sources1 = self._prototype_based_learning(concept_name, concept_type, examples)
            results.append((success1, conf1, sources1))
            
            # Similaridade
            success2, conf2, sources2 = self._similarity_based_learning(concept_name, concept_type, examples)
            results.append((success2, conf2, sources2))
            
            # Transferência
            success3, conf3, sources3 = self._transfer_learning(concept_name, concept_type, examples)
            results.append((success3, conf3, sources3))
            
            # Fusão adaptativa dos resultados
            successful_results = [r for r in results if r[0]]
            
            if not successful_results:
                return False, 0.0, []
            
            # Calcular confiança média dos resultados bem-sucedidos
            avg_confidence = np.mean([r[1] for r in successful_results])
            
            # Combinar fontes de transferência
            all_sources = []
            for _, _, sources in successful_results:
                all_sources.extend(sources)
            
            unique_sources = list(set(all_sources))
            
            return True, avg_confidence, unique_sources
            
        except Exception as e:
            logger.error(f"Erro na fusão adaptativa: {e}")
            return False, 0.0, []
    
    def _get_best_strategy_for_concept_type(self, concept_type: ConceptType) -> FewShotStrategy:
        """Determina a melhor estratégia para um tipo de conceito."""
        # Análise simples baseada no histórico
        if concept_type == ConceptType.SPECIES:
            return FewShotStrategy.PROTOTYPE_BASED
        elif concept_type == ConceptType.BEHAVIOR:
            return FewShotStrategy.SIMILARITY_BASED
        elif concept_type == ConceptType.PATTERN:
            return FewShotStrategy.TRANSFER_LEARNING
        else:
            return FewShotStrategy.ADAPTIVE_FUSION
    
    def _calculate_example_confidence(self, detection_data: Dict[str, Any]) -> float:
        """Calcula confiança de um exemplo baseado nos dados de detecção."""
        try:
            confidences = []
            
            # Extrair confianças de diferentes análises
            for analysis_type in ["shape_analysis", "color_analysis", "texture_analysis", 
                                "pattern_analysis", "behavior_analysis", "motion_analysis"]:
                if analysis_type in detection_data:
                    conf = detection_data[analysis_type].get("confidence", 0.0)
                    confidences.append(conf)
            
            # Retornar confiança média
            return np.mean(confidences) if confidences else 0.5
            
        except Exception as e:
            logger.error(f"Erro ao calcular confiança do exemplo: {e}")
            return 0.5
    
    def _calculate_performance_improvement(self, success: bool, confidence: float) -> float:
        """Calcula melhoria de performance baseada no sucesso."""
        if not success:
            return 0.0
        
        # Melhoria baseada na confiança
        base_improvement = 0.1
        confidence_bonus = confidence * 0.2
        
        return base_improvement + confidence_bonus
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/few_shot_learning.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar base de conhecimento
                if "concept_knowledge_base" in data:
                    self.concept_knowledge_base = data["concept_knowledge_base"]
                
                # Carregar histórico de aprendizado
                if "learning_history" in data:
                    self.learning_history = data["learning_history"]
                
                # Carregar performance das estratégias
                if "strategy_performance" in data:
                    self.strategy_performance = data["strategy_performance"]
                
                logger.info(f"Dados de few-shot learning carregados: {len(self.concept_knowledge_base)} conceitos")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de few-shot learning: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/few_shot_learning.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "concept_knowledge_base": self.concept_knowledge_base,
                "learning_history": [{
                    "concept_name": result.concept_name,
                    "success": result.success,
                    "confidence": result.confidence,
                    "learning_mode": result.learning_mode.value,
                    "strategy_used": result.strategy_used.value,
                    "examples_used": result.examples_used,
                    "transfer_sources": result.transfer_sources,
                    "adaptation_required": result.adaptation_required,
                    "performance_improvement": result.performance_improvement,
                    "timestamp": result.timestamp
                } for result in self.learning_history],
                "strategy_performance": {strategy.value: performances 
                                       for strategy, performances in self.strategy_performance.items()},
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de few-shot learning salvos: {len(self.concept_knowledge_base)} conceitos")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de few-shot learning: {e}")
