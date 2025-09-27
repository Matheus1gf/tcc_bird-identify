#!/usr/bin/env python3
"""
Sistema de Generalização Universal com Transferência entre Espécies
================================================================

Este módulo implementa um sistema avançado de generalização universal que permite:
- Análise de padrões universais entre espécies
- Transferência de conhecimento entre espécies
- Generalização adaptativa baseada em similaridade
- Aprendizado de características universais
- Sistema de mapeamento de características
- Sistema de validação cruzada entre espécies
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UniversalPatternType(Enum):
    """Tipos de padrões universais."""
    MORPHOLOGICAL = "morphological"  # Padrões morfológicos
    BEHAVIORAL = "behavioral"        # Padrões comportamentais
    ECOLOGICAL = "ecological"        # Padrões ecológicos
    PHYSIOLOGICAL = "physiological"  # Padrões fisiológicos
    EVOLUTIONARY = "evolutionary"    # Padrões evolutivos

class TransferStrategy(Enum):
    """Estratégias de transferência de conhecimento."""
    DIRECT = "direct"                # Transferência direta
    ADAPTIVE = "adaptive"            # Transferência adaptativa
    SELECTIVE = "selective"          # Transferência seletiva
    HIERARCHICAL = "hierarchical"    # Transferência hierárquica
    CONTEXTUAL = "contextual"        # Transferência contextual

class GeneralizationLevel(Enum):
    """Níveis de generalização."""
    SPECIES_LEVEL = "species_level"     # Nível de espécie
    GENUS_LEVEL = "genus_level"         # Nível de gênero
    FAMILY_LEVEL = "family_level"       # Nível de família
    ORDER_LEVEL = "order_level"         # Nível de ordem
    CLASS_LEVEL = "class_level"         # Nível de classe
    KINGDOM_LEVEL = "kingdom_level"     # Nível de reino
    UNIVERSAL_LEVEL = "universal_level" # Nível universal

@dataclass
class UniversalPattern:
    """Representa um padrão universal."""
    name: str
    pattern_type: UniversalPatternType
    generalization_level: GeneralizationLevel
    description: str
    characteristics: List[str]
    confidence: float
    transfer_potential: float
    species_applicability: List[str]
    learning_history: List[Dict[str, Any]]
    last_updated: float

@dataclass
class SpeciesMapping:
    """Mapeamento entre espécies."""
    source_species: str
    target_species: str
    similarity_score: float
    transferable_features: List[str]
    transfer_strategy: TransferStrategy
    confidence: float
    success_rate: float
    last_transfer: float

@dataclass
class TransferResult:
    """Resultado de uma transferência de conhecimento."""
    source_species: str
    target_species: str
    transferred_features: List[str]
    success: bool
    confidence: float
    adaptation_required: bool
    performance_improvement: float
    timestamp: float

class UniversalPatternAnalyzer:
    """Analisador de padrões universais."""
    
    def __init__(self):
        self.patterns: Dict[str, UniversalPattern] = {}
        self.species_data: Dict[str, Dict[str, Any]] = {}
        self.pattern_frequency: Counter = Counter()
        self.species_similarity: Dict[Tuple[str, str], float] = {}
        
    def analyze_universal_patterns(self, detection_data: Dict[str, Any], species: str) -> Dict[str, Any]:
        """Analisa padrões universais em dados de detecção."""
        try:
            # Extrair características universais
            universal_features = self._extract_universal_features(detection_data)
            
            # Identificar padrões morfológicos
            morphological_patterns = self._identify_morphological_patterns(universal_features)
            
            # Identificar padrões comportamentais
            behavioral_patterns = self._identify_behavioral_patterns(universal_features)
            
            # Identificar padrões ecológicos
            ecological_patterns = self._identify_ecological_patterns(universal_features)
            
            # Calcular score de universalidade
            universality_score = self._calculate_universality_score(universal_features)
            
            # Calcular potencial de transferência
            transfer_potential = self._calculate_transfer_potential(universal_features, species)
            
            # Identificar características transferíveis
            transferable_features = self._identify_transferable_features(universal_features)
            
            # Atualizar dados da espécie
            self.species_data[species] = {
                "universal_features": universal_features,
                "morphological_patterns": morphological_patterns,
                "behavioral_patterns": behavioral_patterns,
                "ecological_patterns": ecological_patterns,
                "universality_score": universality_score,
                "transfer_potential": transfer_potential,
                "transferable_features": transferable_features,
                "last_analysis": time.time()
            }
            
            return {
                "universal_features": universal_features,
                "morphological_patterns": morphological_patterns,
                "behavioral_patterns": behavioral_patterns,
                "ecological_patterns": ecological_patterns,
                "universality_score": universality_score,
                "transfer_potential": transfer_potential,
                "transferable_features": transferable_features,
                "species": species,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erro ao analisar padrões universais: {e}")
            return {"error": str(e)}
    
    def _extract_universal_features(self, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai características universais dos dados de detecção."""
        features = {}
        
        # Características morfológicas universais
        if "shape_analysis" in detection_data:
            features["bilateral_symmetry"] = detection_data["shape_analysis"].get("symmetrical", False)
            features["compact_body"] = "compact" in detection_data["shape_analysis"].get("detected_shapes", [])
            features["head_structure"] = "head" in detection_data["shape_analysis"].get("detected_shapes", [])
        
        # Características comportamentais universais
        if "behavior_analysis" in detection_data:
            features["feeding_behavior"] = detection_data["behavior_analysis"].get("feeding_behavior", False)
            features["movement_detected"] = detection_data["motion_analysis"].get("movement_detected", False)
        
        # Características ecológicas universais
        if "position_analysis" in detection_data:
            features["terrestrial_position"] = "ground" in detection_data["position_analysis"].get("detected_positions", [])
            features["aerial_position"] = "air" in detection_data["position_analysis"].get("detected_positions", [])
        
        return features
    
    def _identify_morphological_patterns(self, features: Dict[str, Any]) -> List[str]:
        """Identifica padrões morfológicos universais."""
        patterns = []
        
        if features.get("bilateral_symmetry"):
            patterns.append("bilateral_symmetry")
        if features.get("compact_body"):
            patterns.append("compact_body_structure")
        if features.get("head_structure"):
            patterns.append("cephalic_structure")
        
        return patterns
    
    def _identify_behavioral_patterns(self, features: Dict[str, Any]) -> List[str]:
        """Identifica padrões comportamentais universais."""
        patterns = []
        
        if features.get("feeding_behavior"):
            patterns.append("feeding_behavior")
        if features.get("movement_detected"):
            patterns.append("locomotion")
        
        return patterns
    
    def _identify_ecological_patterns(self, features: Dict[str, Any]) -> List[str]:
        """Identifica padrões ecológicos universais."""
        patterns = []
        
        if features.get("terrestrial_position"):
            patterns.append("terrestrial_habitat")
        if features.get("aerial_position"):
            patterns.append("aerial_habitat")
        
        return patterns
    
    def _calculate_universality_score(self, features: Dict[str, Any]) -> float:
        """Calcula score de universalidade das características."""
        if not features:
            return 0.0
        
        # Características universais conhecidas
        universal_characteristics = {
            "bilateral_symmetry": 0.9,
            "compact_body": 0.7,
            "head_structure": 0.8,
            "feeding_behavior": 0.6,
            "movement_detected": 0.5,
            "terrestrial_position": 0.4,
            "aerial_position": 0.3
        }
        
        total_score = 0.0
        total_weight = 0.0
        
        for feature, detected in features.items():
            if detected and feature in universal_characteristics:
                weight = universal_characteristics[feature]
                total_score += weight
                total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_transfer_potential(self, features: Dict[str, Any], species: str) -> float:
        """Calcula potencial de transferência das características."""
        if not features:
            return 0.0
        
        # Características com alto potencial de transferência
        high_transfer_features = {
            "bilateral_symmetry": 0.9,
            "compact_body": 0.8,
            "head_structure": 0.7,
            "feeding_behavior": 0.6,
            "movement_detected": 0.5
        }
        
        total_potential = 0.0
        total_weight = 0.0
        
        for feature, detected in features.items():
            if detected and feature in high_transfer_features:
                weight = high_transfer_features[feature]
                total_potential += weight
                total_weight += weight
        
        return total_potential / total_weight if total_weight > 0 else 0.0
    
    def _identify_transferable_features(self, features: Dict[str, Any]) -> List[str]:
        """Identifica características transferíveis."""
        transferable = []
        
        for feature, detected in features.items():
            if detected and feature in ["bilateral_symmetry", "compact_body", "head_structure", "feeding_behavior"]:
                transferable.append(feature)
        
        return transferable

class SpeciesTransferSystem:
    """Sistema de transferência entre espécies."""
    
    def __init__(self):
        self.species_mappings: Dict[Tuple[str, str], SpeciesMapping] = {}
        self.transfer_history: List[TransferResult] = []
        self.species_similarity_matrix: Dict[Tuple[str, str], float] = {}
        self.transfer_strategies: Dict[TransferStrategy, float] = {
            TransferStrategy.DIRECT: 0.8,
            TransferStrategy.ADAPTIVE: 0.9,
            TransferStrategy.SELECTIVE: 0.7,
            TransferStrategy.HIERARCHICAL: 0.85,
            TransferStrategy.CONTEXTUAL: 0.75
        }
    
    def transfer_knowledge(self, source_species: str, target_species: str, 
                          features: List[str], strategy: TransferStrategy = TransferStrategy.ADAPTIVE) -> TransferResult:
        """Transfere conhecimento entre espécies."""
        try:
            # Calcular similaridade entre espécies
            similarity = self._calculate_species_similarity(source_species, target_species)
            
            # Determinar características transferíveis
            transferable_features = self._filter_transferable_features(features, similarity)
            
            # Aplicar estratégia de transferência
            if strategy == TransferStrategy.DIRECT:
                success, confidence = self._direct_transfer(source_species, target_species, transferable_features)
            elif strategy == TransferStrategy.ADAPTIVE:
                success, confidence = self._adaptive_transfer(source_species, target_species, transferable_features)
            elif strategy == TransferStrategy.SELECTIVE:
                success, confidence = self._selective_transfer(source_species, target_species, transferable_features)
            elif strategy == TransferStrategy.HIERARCHICAL:
                success, confidence = self._hierarchical_transfer(source_species, target_species, transferable_features)
            elif strategy == TransferStrategy.CONTEXTUAL:
                success, confidence = self._contextual_transfer(source_species, target_species, transferable_features)
            else:
                success, confidence = False, 0.0
            
            # Calcular melhoria de performance
            performance_improvement = self._calculate_performance_improvement(success, confidence)
            
            # Criar resultado da transferência
            result = TransferResult(
                source_species=source_species,
                target_species=target_species,
                transferred_features=transferable_features,
                success=success,
                confidence=confidence,
                adaptation_required=strategy != TransferStrategy.DIRECT,
                performance_improvement=performance_improvement,
                timestamp=time.time()
            )
            
            # Atualizar histórico
            self.transfer_history.append(result)
            
            # Atualizar mapeamento de espécies
            mapping_key = (source_species, target_species)
            if mapping_key not in self.species_mappings:
                self.species_mappings[mapping_key] = SpeciesMapping(
                    source_species=source_species,
                    target_species=target_species,
                    similarity_score=similarity,
                    transferable_features=transferable_features,
                    transfer_strategy=strategy,
                    confidence=confidence,
                    success_rate=0.0,
                    last_transfer=time.time()
                )
            
            # Atualizar taxa de sucesso
            self.species_mappings[mapping_key].success_rate = self._calculate_success_rate(mapping_key)
            
            return result
            
        except Exception as e:
            logger.error(f"Erro ao transferir conhecimento: {e}")
            return TransferResult(
                source_species=source_species,
                target_species=target_species,
                transferred_features=[],
                success=False,
                confidence=0.0,
                adaptation_required=False,
                performance_improvement=0.0,
                timestamp=time.time()
            )
    
    def _calculate_species_similarity(self, species1: str, species2: str) -> float:
        """Calcula similaridade entre duas espécies."""
        # Similaridade baseada em características conhecidas
        similarity_matrix = {
            ("bird", "mammal"): 0.3,
            ("bird", "reptile"): 0.4,
            ("bird", "fish"): 0.2,
            ("mammal", "reptile"): 0.5,
            ("mammal", "fish"): 0.2,
            ("reptile", "fish"): 0.3
        }
        
        # Verificar se já temos a similaridade calculada
        key1 = (species1, species2)
        key2 = (species2, species1)
        
        if key1 in similarity_matrix:
            return similarity_matrix[key1]
        elif key2 in similarity_matrix:
            return similarity_matrix[key2]
        else:
            # Similaridade padrão baseada em características universais
            return 0.5
    
    def _filter_transferable_features(self, features: List[str], similarity: float) -> List[str]:
        """Filtra características transferíveis baseado na similaridade."""
        if similarity < 0.3:
            return []
        elif similarity < 0.5:
            return [f for f in features if f in ["bilateral_symmetry", "compact_body"]]
        elif similarity < 0.7:
            return [f for f in features if f in ["bilateral_symmetry", "compact_body", "head_structure"]]
        else:
            return features
    
    def _direct_transfer(self, source_species: str, target_species: str, features: List[str]) -> Tuple[bool, float]:
        """Transferência direta de características."""
        if not features:
            return False, 0.0
        
        # Transferência direta tem alta confiança para características universais
        confidence = 0.8 if "bilateral_symmetry" in features else 0.6
        success = len(features) > 0
        
        return success, confidence
    
    def _adaptive_transfer(self, source_species: str, target_species: str, features: List[str]) -> Tuple[bool, float]:
        """Transferência adaptativa de características."""
        if not features:
            return False, 0.0
        
        # Transferência adaptativa ajusta características baseado na espécie alvo
        adapted_features = self._adapt_features_to_species(features, target_species)
        confidence = 0.9 if len(adapted_features) > 0 else 0.0
        success = len(adapted_features) > 0
        
        return success, confidence
    
    def _selective_transfer(self, source_species: str, target_species: str, features: List[str]) -> Tuple[bool, float]:
        """Transferência seletiva de características."""
        if not features:
            return False, 0.0
        
        # Seleciona apenas características mais relevantes
        selected_features = [f for f in features if f in ["bilateral_symmetry", "compact_body"]]
        confidence = 0.7 if len(selected_features) > 0 else 0.0
        success = len(selected_features) > 0
        
        return success, confidence
    
    def _hierarchical_transfer(self, source_species: str, target_species: str, features: List[str]) -> Tuple[bool, float]:
        """Transferência hierárquica de características."""
        if not features:
            return False, 0.0
        
        # Transferência hierárquica considera níveis de abstração
        hierarchical_features = self._apply_hierarchical_filtering(features, source_species, target_species)
        confidence = 0.85 if len(hierarchical_features) > 0 else 0.0
        success = len(hierarchical_features) > 0
        
        return success, confidence
    
    def _contextual_transfer(self, source_species: str, target_species: str, features: List[str]) -> Tuple[bool, float]:
        """Transferência contextual de características."""
        if not features:
            return False, 0.0
        
        # Transferência contextual considera contexto específico
        contextual_features = self._apply_contextual_filtering(features, source_species, target_species)
        confidence = 0.75 if len(contextual_features) > 0 else 0.0
        success = len(contextual_features) > 0
        
        return success, confidence
    
    def _adapt_features_to_species(self, features: List[str], target_species: str) -> List[str]:
        """Adapta características para a espécie alvo."""
        adapted = []
        
        for feature in features:
            if feature == "bilateral_symmetry":
                adapted.append(feature)  # Universal
            elif feature == "compact_body" and target_species in ["bird", "mammal"]:
                adapted.append(feature)  # Aplicável a aves e mamíferos
            elif feature == "head_structure" and target_species in ["bird", "mammal", "reptile"]:
                adapted.append(feature)  # Aplicável a vertebrados
        
        return adapted
    
    def _apply_hierarchical_filtering(self, features: List[str], source_species: str, target_species: str) -> List[str]:
        """Aplica filtragem hierárquica baseada em níveis de abstração."""
        # Características universais (nível mais alto)
        universal_features = ["bilateral_symmetry"]
        
        # Características de classe (nível médio)
        class_features = ["compact_body", "head_structure"]
        
        # Características específicas (nível baixo)
        specific_features = ["feeding_behavior", "movement_detected"]
        
        # Determinar nível de transferência baseado na similaridade
        similarity = self._calculate_species_similarity(source_species, target_species)
        
        if similarity >= 0.7:
            return universal_features + class_features + specific_features
        elif similarity >= 0.5:
            return universal_features + class_features
        else:
            return universal_features
    
    def _apply_contextual_filtering(self, features: List[str], source_species: str, target_species: str) -> List[str]:
        """Aplica filtragem contextual baseada no contexto específico."""
        contextual = []
        
        # Contexto de habitat
        if source_species == "bird" and target_species == "mammal":
            contextual.extend([f for f in features if f in ["bilateral_symmetry", "compact_body"]])
        elif source_species == "mammal" and target_species == "bird":
            contextual.extend([f for f in features if f in ["bilateral_symmetry", "head_structure"]])
        else:
            contextual.extend([f for f in features if f in ["bilateral_symmetry"]])
        
        return contextual
    
    def _calculate_performance_improvement(self, success: bool, confidence: float) -> float:
        """Calcula melhoria de performance baseada no sucesso da transferência."""
        if not success:
            return 0.0
        
        # Melhoria baseada na confiança e sucesso
        base_improvement = 0.1
        confidence_bonus = confidence * 0.2
        
        return base_improvement + confidence_bonus
    
    def _calculate_success_rate(self, mapping_key: Tuple[str, str]) -> float:
        """Calcula taxa de sucesso para um mapeamento de espécies."""
        relevant_transfers = [t for t in self.transfer_history 
                             if (t.source_species, t.target_species) == mapping_key]
        
        if not relevant_transfers:
            return 0.0
        
        successful_transfers = sum(1 for t in relevant_transfers if t.success)
        return successful_transfers / len(relevant_transfers)

class UniversalGeneralizationSystem:
    """Sistema principal de generalização universal."""
    
    def __init__(self):
        self.pattern_analyzer = UniversalPatternAnalyzer()
        self.transfer_system = SpeciesTransferSystem()
        self.generalization_history: List[Dict[str, Any]] = []
        self.species_knowledge_base: Dict[str, Dict[str, Any]] = {}
        self.universal_patterns: Dict[str, UniversalPattern] = {}
        
        # Carregar dados existentes
        self._load_data()
    
    def analyze_for_universal_generalization(self, detection_data: Dict[str, Any], 
                                           species: str, context: str = "general") -> Dict[str, Any]:
        """Analisa dados para generalização universal."""
        try:
            # Análise de padrões universais
            pattern_analysis = self.pattern_analyzer.analyze_universal_patterns(detection_data, species)
            
            if "error" in pattern_analysis:
                return pattern_analysis
            
            # Identificar espécies similares para transferência
            similar_species = self._find_similar_species(species)
            
            # Calcular potencial de generalização
            generalization_potential = self._calculate_generalization_potential(pattern_analysis)
            
            # Identificar características universais
            universal_characteristics = self._identify_universal_characteristics(pattern_analysis)
            
            # Sugerir transferências
            transfer_suggestions = self._suggest_transfers(species, similar_species, universal_characteristics)
            
            # Calcular score de generalização universal
            universal_generalization_score = self._calculate_universal_generalization_score(
                pattern_analysis, universal_characteristics, transfer_suggestions
            )
            
            # Atualizar base de conhecimento
            self.species_knowledge_base[species] = {
                "pattern_analysis": pattern_analysis,
                "similar_species": similar_species,
                "generalization_potential": generalization_potential,
                "universal_characteristics": universal_characteristics,
                "transfer_suggestions": transfer_suggestions,
                "universal_generalization_score": universal_generalization_score,
                "last_analysis": time.time()
            }
            
            return {
                "pattern_analysis": pattern_analysis,
                "similar_species": similar_species,
                "generalization_potential": generalization_potential,
                "universal_characteristics": universal_characteristics,
                "transfer_suggestions": transfer_suggestions,
                "universal_generalization_score": universal_generalization_score,
                "species": species,
                "context": context,
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de generalização universal: {e}")
            return {"error": str(e)}
    
    def transfer_knowledge_between_species(self, source_species: str, target_species: str, 
                                         features: List[str], strategy: TransferStrategy = TransferStrategy.ADAPTIVE) -> Dict[str, Any]:
        """Transfere conhecimento entre espécies."""
        try:
            # Realizar transferência
            transfer_result = self.transfer_system.transfer_knowledge(
                source_species, target_species, features, strategy
            )
            
            # Atualizar histórico de generalização
            self.generalization_history.append({
                "type": "knowledge_transfer",
                "source_species": source_species,
                "target_species": target_species,
                "features": features,
                "strategy": strategy.value,
                "result": asdict(transfer_result),
                "timestamp": time.time()
            })
            
            # Atualizar padrões universais
            self._update_universal_patterns(transfer_result)
            
            return {
                "transfer_result": asdict(transfer_result),
                "success": transfer_result.success,
                "confidence": transfer_result.confidence,
                "performance_improvement": transfer_result.performance_improvement,
                "adaptation_required": transfer_result.adaptation_required
            }
            
        except Exception as e:
            logger.error(f"Erro na transferência de conhecimento: {e}")
            return {"error": str(e)}
    
    def learn_universal_pattern(self, detection_data: Dict[str, Any], species: str, 
                               pattern_name: str, pattern_type: UniversalPatternType) -> Dict[str, Any]:
        """Aprende um novo padrão universal."""
        try:
            # Analisar padrões universais
            pattern_analysis = self.pattern_analyzer.analyze_universal_patterns(detection_data, species)
            
            if "error" in pattern_analysis:
                return pattern_analysis
            
            # Criar padrão universal
            universal_pattern = UniversalPattern(
                name=pattern_name,
                pattern_type=pattern_type,
                generalization_level=GeneralizationLevel.SPECIES_LEVEL,
                description=f"Padrão universal aprendido de {species}",
                characteristics=pattern_analysis.get("transferable_features", []),
                confidence=pattern_analysis.get("universality_score", 0.0),
                transfer_potential=pattern_analysis.get("transfer_potential", 0.0),
                species_applicability=[species],
                learning_history=[{
                    "species": species,
                    "detection_data": detection_data,
                    "timestamp": time.time()
                }],
                last_updated=time.time()
            )
            
            # Adicionar à base de padrões universais
            self.universal_patterns[pattern_name] = universal_pattern
            
            # Atualizar histórico
            self.generalization_history.append({
                "type": "pattern_learning",
                "pattern_name": pattern_name,
                "pattern_type": pattern_type.value,
                "species": species,
                "pattern": asdict(universal_pattern),
                "timestamp": time.time()
            })
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "pattern_name": pattern_name,
                "pattern_type": pattern_type.value,
                "species": species,
                "confidence": universal_pattern.confidence,
                "transfer_potential": universal_pattern.transfer_potential,
                "characteristics": universal_pattern.characteristics
            }
            
        except Exception as e:
            logger.error(f"Erro ao aprender padrão universal: {e}")
            return {"error": str(e)}
    
    def get_universal_generalization_analysis(self, species: str = None) -> Dict[str, Any]:
        """Obtém análise de generalização universal."""
        try:
            if species:
                # Análise específica de uma espécie
                if species in self.species_knowledge_base:
                    return self.species_knowledge_base[species]
                else:
                    return {"error": f"Espécie {species} não encontrada na base de conhecimento"}
            
            # Análise geral
            total_species = len(self.species_knowledge_base)
            total_patterns = len(self.universal_patterns)
            total_transfers = len(self.transfer_system.transfer_history)
            
            # Calcular métricas de generalização
            average_generalization_score = np.mean([
                data.get("universal_generalization_score", 0.0) 
                for data in self.species_knowledge_base.values()
            ]) if self.species_knowledge_base else 0.0
            
            average_transfer_success = np.mean([
                t.success for t in self.transfer_system.transfer_history
            ]) if self.transfer_system.transfer_history else 0.0
            
            # Espécies com maior potencial de generalização
            species_by_potential = sorted(
                self.species_knowledge_base.items(),
                key=lambda x: x[1].get("generalization_potential", 0.0),
                reverse=True
            )
            
            return {
                "total_species": total_species,
                "total_patterns": total_patterns,
                "total_transfers": total_transfers,
                "average_generalization_score": average_generalization_score,
                "average_transfer_success": average_transfer_success,
                "species_by_potential": species_by_potential[:5],  # Top 5
                "universal_patterns": {name: {
                    "name": pattern.name,
                    "pattern_type": pattern.pattern_type.value,
                    "generalization_level": pattern.generalization_level.value,
                    "description": pattern.description,
                    "characteristics": pattern.characteristics,
                    "confidence": pattern.confidence,
                    "transfer_potential": pattern.transfer_potential,
                    "species_applicability": pattern.species_applicability,
                    "learning_history": pattern.learning_history,
                    "last_updated": pattern.last_updated
                } for name, pattern in self.universal_patterns.items()},
                "transfer_history": [asdict(t) for t in self.transfer_system.transfer_history[-10:]],  # Últimos 10
                "analysis_timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de generalização universal: {e}")
            return {"error": str(e)}
    
    def _find_similar_species(self, species: str) -> List[str]:
        """Encontra espécies similares para transferência."""
        # Espécies conhecidas e suas similaridades
        species_similarities = {
            "bird": ["mammal", "reptile"],
            "mammal": ["bird", "reptile"],
            "reptile": ["bird", "mammal"],
            "fish": ["reptile"],
            "insect": ["spider"]
        }
        
        return species_similarities.get(species, [])
    
    def _calculate_generalization_potential(self, pattern_analysis: Dict[str, Any]) -> float:
        """Calcula potencial de generalização."""
        universality_score = pattern_analysis.get("universality_score", 0.0)
        transfer_potential = pattern_analysis.get("transfer_potential", 0.0)
        
        # Potencial baseado na universalidade e transferência
        return (universality_score + transfer_potential) / 2.0
    
    def _identify_universal_characteristics(self, pattern_analysis: Dict[str, Any]) -> List[str]:
        """Identifica características universais."""
        universal_features = pattern_analysis.get("universal_features", {})
        transferable_features = pattern_analysis.get("transferable_features", [])
        
        # Características universais são aquelas com alta universalidade e transferibilidade
        universal_characteristics = []
        
        for feature in transferable_features:
            if feature in universal_features and universal_features[feature]:
                universal_characteristics.append(feature)
        
        return universal_characteristics
    
    def _suggest_transfers(self, species: str, similar_species: List[str], 
                          universal_characteristics: List[str]) -> List[Dict[str, Any]]:
        """Sugere transferências de conhecimento."""
        suggestions = []
        
        for target_species in similar_species:
            # Calcular similaridade
            similarity = self.transfer_system._calculate_species_similarity(species, target_species)
            
            # Determinar estratégia de transferência
            if similarity >= 0.7:
                strategy = TransferStrategy.DIRECT
            elif similarity >= 0.5:
                strategy = TransferStrategy.ADAPTIVE
            else:
                strategy = TransferStrategy.SELECTIVE
            
            # Criar sugestão
            suggestion = {
                "target_species": target_species,
                "similarity": similarity,
                "strategy": strategy.value,
                "features": universal_characteristics,
                "confidence": similarity * 0.8
            }
            
            suggestions.append(suggestion)
        
        return suggestions
    
    def _calculate_universal_generalization_score(self, pattern_analysis: Dict[str, Any], 
                                                universal_characteristics: List[str], 
                                                transfer_suggestions: List[Dict[str, Any]]) -> float:
        """Calcula score de generalização universal."""
        # Score baseado em múltiplos fatores
        universality_score = pattern_analysis.get("universality_score", 0.0)
        transfer_potential = pattern_analysis.get("transfer_potential", 0.0)
        characteristics_count = len(universal_characteristics)
        suggestions_count = len(transfer_suggestions)
        
        # Fórmula de score universal
        base_score = (universality_score + transfer_potential) / 2.0
        characteristics_bonus = characteristics_count * 0.1
        suggestions_bonus = suggestions_count * 0.05
        
        return min(base_score + characteristics_bonus + suggestions_bonus, 1.0)
    
    def _update_universal_patterns(self, transfer_result: TransferResult):
        """Atualiza padrões universais baseado no resultado da transferência."""
        if not transfer_result.success:
            return
        
        # Atualizar padrões existentes
        for pattern_name, pattern in self.universal_patterns.items():
            if transfer_result.source_species in pattern.species_applicability:
                # Adicionar espécie alvo se não estiver presente
                if transfer_result.target_species not in pattern.species_applicability:
                    pattern.species_applicability.append(transfer_result.target_species)
                
                # Atualizar confiança baseada no sucesso da transferência
                pattern.confidence = min(pattern.confidence + 0.05, 1.0)
                pattern.last_updated = time.time()
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/universal_generalization.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar padrões universais
                if "universal_patterns" in data:
                    for name, pattern_data in data["universal_patterns"].items():
                        self.universal_patterns[name] = UniversalPattern(**pattern_data)
                
                # Carregar base de conhecimento de espécies
                if "species_knowledge_base" in data:
                    self.species_knowledge_base = data["species_knowledge_base"]
                
                # Carregar histórico de generalização
                if "generalization_history" in data:
                    self.generalization_history = data["generalization_history"]
                
                logger.info(f"Dados de generalização universal carregados: {len(self.universal_patterns)} padrões, {len(self.species_knowledge_base)} espécies")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de generalização universal: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/universal_generalization.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "universal_patterns": {name: {
                    "name": pattern.name,
                    "pattern_type": pattern.pattern_type.value,
                    "generalization_level": pattern.generalization_level.value,
                    "description": pattern.description,
                    "characteristics": pattern.characteristics,
                    "confidence": pattern.confidence,
                    "transfer_potential": pattern.transfer_potential,
                    "species_applicability": pattern.species_applicability,
                    "learning_history": pattern.learning_history,
                    "last_updated": pattern.last_updated
                } for name, pattern in self.universal_patterns.items()},
                "species_knowledge_base": self.species_knowledge_base,
                "generalization_history": self.generalization_history,
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de generalização universal salvos: {len(self.universal_patterns)} padrões, {len(self.species_knowledge_base)} espécies")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de generalização universal: {e}")
