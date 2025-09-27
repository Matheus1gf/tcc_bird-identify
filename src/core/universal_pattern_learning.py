#!/usr/bin/env python3
"""
Sistema de Aprendizado por Padrões Universais
============================================

Este módulo implementa um sistema avançado de aprendizado por padrões universais que permite:
- Identificação de padrões universais em dados de detecção
- Aprendizado de características comuns entre espécies
- Generalização de padrões para novas espécies
- Sistema de validação de padrões universais
- Transferência de conhecimento baseada em padrões
- Sistema de hierarquia de padrões universais
- Aprendizado incremental de padrões
- Sistema de confiança para padrões universais
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

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PatternType(Enum):
    """Tipos de padrões universais."""
    MORPHOLOGICAL = "morphological"     # Morfológico (forma, estrutura)
    BEHAVIORAL = "behavioral"           # Comportamental (comportamento)
    ECOLOGICAL = "ecological"           # Ecológico (habitat, nicho)
    PHYSIOLOGICAL = "physiological"     # Fisiológico (funções corporais)
    EVOLUTIONARY = "evolutionary"       # Evolutivo (adaptações)
    FUNCTIONAL = "functional"           # Funcional (funções)

class PatternLevel(Enum):
    """Níveis de abstração dos padrões."""
    SPECIES_LEVEL = "species_level"     # Nível de espécie
    GENUS_LEVEL = "genus_level"         # Nível de gênero
    FAMILY_LEVEL = "family_level"      # Nível de família
    ORDER_LEVEL = "order_level"        # Nível de ordem
    CLASS_LEVEL = "class_level"         # Nível de classe
    PHYLUM_LEVEL = "phylum_level"       # Nível de filo
    KINGDOM_LEVEL = "kingdom_level"     # Nível de reino

class PatternConfidence(Enum):
    """Níveis de confiança dos padrões."""
    LOW = "low"                         # Baixa (0.0 - 0.3)
    MEDIUM = "medium"                   # Média (0.3 - 0.6)
    HIGH = "high"                       # Alta (0.6 - 0.8)
    VERY_HIGH = "very_high"             # Muito alta (0.8 - 1.0)

@dataclass
class UniversalPattern:
    """Representa um padrão universal."""
    pattern_id: str
    name: str
    pattern_type: PatternType
    pattern_level: PatternLevel
    description: str
    characteristics: List[str]
    applicable_species: List[str]
    confidence: float
    confidence_level: PatternConfidence
    evidence_count: int
    validation_score: float
    learning_history: List[Dict[str, Any]]
    created_at: float
    last_updated: float

@dataclass
class PatternLearningResult:
    """Resultado do aprendizado de padrão."""
    result_id: str
    pattern_id: str
    species: str
    learning_method: str
    confidence_gained: float
    new_characteristics: List[str]
    validation_passed: bool
    timestamp: float

class UniversalPatternLearner:
    """Sistema de aprendizado por padrões universais."""
    
    def __init__(self):
        self.universal_patterns: Dict[str, UniversalPattern] = {}
        self.learning_results: List[PatternLearningResult] = []
        self.species_patterns: Dict[str, List[str]] = defaultdict(list)
        self.pattern_hierarchy: Dict[str, List[str]] = defaultdict(list)
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar padrões básicos
        self._initialize_basic_patterns()
    
    def learn_universal_pattern(self, detection_data: Dict[str, Any], 
                              species: str, pattern_name: str = None) -> Dict[str, Any]:
        """Aprende um padrão universal a partir de dados de detecção."""
        try:
            # Extrair características dos dados de detecção
            characteristics = self._extract_characteristics(detection_data)
            
            # Determinar tipo de padrão
            pattern_type = self._determine_pattern_type(characteristics)
            
            # Determinar nível de abstração
            pattern_level = self._determine_pattern_level(species, characteristics)
            
            # Gerar nome do padrão se não fornecido
            if not pattern_name:
                pattern_name = self._generate_pattern_name(pattern_type, pattern_level)
            
            # Calcular confiança inicial
            confidence = self._calculate_initial_confidence(characteristics, species)
            
            # Determinar nível de confiança
            confidence_level = self._determine_confidence_level(confidence)
            
            # Criar padrão universal
            pattern_id = f"pattern_{int(time.time())}_{random.randint(1000, 9999)}"
            pattern = UniversalPattern(
                pattern_id=pattern_id,
                name=pattern_name,
                pattern_type=pattern_type,
                pattern_level=pattern_level,
                description=f"Padrão universal {pattern_type.value} identificado em {species}",
                characteristics=characteristics,
                applicable_species=[species],
                confidence=confidence,
                confidence_level=confidence_level,
                evidence_count=1,
                validation_score=confidence,
                learning_history=[{
                    "species": species,
                    "method": "initial_learning",
                    "confidence_gained": confidence,
                    "timestamp": time.time()
                }],
                created_at=time.time(),
                last_updated=time.time()
            )
            
            # Adicionar padrão ao sistema
            self.universal_patterns[pattern_id] = pattern
            self.species_patterns[species].append(pattern_id)
            
            # Atualizar hierarquia de padrões
            self._update_pattern_hierarchy(pattern)
            
            # Criar resultado de aprendizado
            learning_result = PatternLearningResult(
                result_id=f"learning_{int(time.time())}_{random.randint(1000, 9999)}",
                pattern_id=pattern_id,
                species=species,
                learning_method="initial_learning",
                confidence_gained=confidence,
                new_characteristics=characteristics,
                validation_passed=True,
                timestamp=time.time()
            )
            
            self.learning_results.append(learning_result)
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "pattern_id": pattern_id,
                "pattern_name": pattern_name,
                "pattern_type": pattern_type.value,
                "pattern_level": pattern_level.value,
                "confidence": confidence,
                "confidence_level": confidence_level.value,
                "characteristics": characteristics,
                "applicable_species": [species],
                "message": f"Padrão universal '{pattern_name}' aprendido com sucesso"
            }
            
        except Exception as e:
            logger.error(f"Erro no aprendizado de padrão universal: {e}")
            return {"error": str(e)}
    
    def _extract_characteristics(self, detection_data: Dict[str, Any]) -> List[str]:
        """Extrai características dos dados de detecção."""
        characteristics = []
        
        try:
            # Extrair características morfológicas
            if "shape" in detection_data:
                shape_data = detection_data["shape"]
                if isinstance(shape_data, dict):
                    for key, value in shape_data.items():
                        if value and isinstance(value, (str, int, float)):
                            characteristics.append(f"shape_{key}_{value}")
            
            # Extrair características de cor
            if "color" in detection_data:
                color_data = detection_data["color"]
                if isinstance(color_data, dict):
                    for key, value in color_data.items():
                        if value and isinstance(value, (str, int, float)):
                            characteristics.append(f"color_{key}_{value}")
            
            # Extrair características de textura
            if "texture" in detection_data:
                texture_data = detection_data["texture"]
                if isinstance(texture_data, dict):
                    for key, value in texture_data.items():
                        if value and isinstance(value, (str, int, float)):
                            characteristics.append(f"texture_{key}_{value}")
            
            # Extrair características comportamentais
            if "behavior" in detection_data:
                behavior_data = detection_data["behavior"]
                if isinstance(behavior_data, dict):
                    for key, value in behavior_data.items():
                        if value and isinstance(value, (str, int, float)):
                            characteristics.append(f"behavior_{key}_{value}")
            
            # Extrair características ecológicas
            if "habitat" in detection_data:
                habitat_data = detection_data["habitat"]
                if isinstance(habitat_data, dict):
                    for key, value in habitat_data.items():
                        if value and isinstance(value, (str, int, float)):
                            characteristics.append(f"habitat_{key}_{value}")
            
            return characteristics
            
        except Exception as e:
            logger.error(f"Erro na extração de características: {e}")
            return []
    
    def _determine_pattern_type(self, characteristics: List[str]) -> PatternType:
        """Determina o tipo de padrão baseado nas características."""
        try:
            # Contar tipos de características
            type_counts = defaultdict(int)
            
            for char in characteristics:
                if char.startswith("shape_") or char.startswith("morphology_"):
                    type_counts["morphological"] += 1
                elif char.startswith("behavior_"):
                    type_counts["behavioral"] += 1
                elif char.startswith("habitat_") or char.startswith("ecological_"):
                    type_counts["ecological"] += 1
                elif char.startswith("physiology_") or char.startswith("function_"):
                    type_counts["physiological"] += 1
                elif char.startswith("evolution_") or char.startswith("adaptation_"):
                    type_counts["evolutionary"] += 1
                elif char.startswith("function_"):
                    type_counts["functional"] += 1
            
            # Retornar tipo mais comum
            if type_counts:
                most_common_type = max(type_counts, key=type_counts.get)
                return PatternType(most_common_type)
            else:
                return PatternType.MORPHOLOGICAL  # Padrão padrão
                
        except Exception as e:
            logger.error(f"Erro na determinação do tipo de padrão: {e}")
            return PatternType.MORPHOLOGICAL
    
    def _determine_pattern_level(self, species: str, characteristics: List[str]) -> PatternLevel:
        """Determina o nível de abstração do padrão."""
        try:
            # Baseado no número de características e complexidade
            char_count = len(characteristics)
            
            if char_count >= 10:
                return PatternLevel.KINGDOM_LEVEL
            elif char_count >= 8:
                return PatternLevel.PHYLUM_LEVEL
            elif char_count >= 6:
                return PatternLevel.CLASS_LEVEL
            elif char_count >= 4:
                return PatternLevel.ORDER_LEVEL
            elif char_count >= 2:
                return PatternLevel.FAMILY_LEVEL
            else:
                return PatternLevel.SPECIES_LEVEL
                
        except Exception as e:
            logger.error(f"Erro na determinação do nível de padrão: {e}")
            return PatternLevel.SPECIES_LEVEL
    
    def _generate_pattern_name(self, pattern_type: PatternType, pattern_level: PatternLevel) -> str:
        """Gera nome para o padrão."""
        try:
            type_names = {
                PatternType.MORPHOLOGICAL: "morfológico",
                PatternType.BEHAVIORAL: "comportamental",
                PatternType.ECOLOGICAL: "ecológico",
                PatternType.PHYSIOLOGICAL: "fisiológico",
                PatternType.EVOLUTIONARY: "evolutivo",
                PatternType.FUNCTIONAL: "funcional"
            }
            
            level_names = {
                PatternLevel.SPECIES_LEVEL: "espécie",
                PatternLevel.GENUS_LEVEL: "gênero",
                PatternLevel.FAMILY_LEVEL: "família",
                PatternLevel.ORDER_LEVEL: "ordem",
                PatternLevel.CLASS_LEVEL: "classe",
                PatternLevel.PHYLUM_LEVEL: "filo",
                PatternLevel.KINGDOM_LEVEL: "reino"
            }
            
            type_name = type_names.get(pattern_type, "universal")
            level_name = level_names.get(pattern_level, "geral")
            
            return f"Padrão {type_name} de {level_name}"
            
        except Exception as e:
            logger.error(f"Erro na geração do nome do padrão: {e}")
            return "Padrão Universal"
    
    def _calculate_initial_confidence(self, characteristics: List[str], species: str) -> float:
        """Calcula confiança inicial do padrão."""
        try:
            # Baseado no número de características e diversidade
            char_count = len(characteristics)
            
            # Confiança baseada no número de características
            count_confidence = min(char_count / 10.0, 1.0)
            
            # Confiança baseada na diversidade de tipos
            type_diversity = len(set(char.split("_")[0] for char in characteristics))
            diversity_confidence = min(type_diversity / 5.0, 1.0)
            
            # Confiança baseada na espécie (espécies conhecidas têm maior confiança)
            species_confidence = 0.8 if species in self.species_patterns else 0.5
            
            # Calcular confiança geral
            overall_confidence = (count_confidence * 0.4 + 
                                diversity_confidence * 0.3 + 
                                species_confidence * 0.3)
            
            return min(overall_confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de confiança inicial: {e}")
            return 0.5
    
    def _determine_confidence_level(self, confidence: float) -> PatternConfidence:
        """Determina nível de confiança."""
        if confidence >= 0.8:
            return PatternConfidence.VERY_HIGH
        elif confidence >= 0.6:
            return PatternConfidence.HIGH
        elif confidence >= 0.3:
            return PatternConfidence.MEDIUM
        else:
            return PatternConfidence.LOW
    
    def _update_pattern_hierarchy(self, pattern: UniversalPattern):
        """Atualiza hierarquia de padrões."""
        try:
            # Adicionar padrão à hierarquia por tipo
            pattern_type = pattern.pattern_type.value
            self.pattern_hierarchy[pattern_type].append(pattern.pattern_id)
            
            # Adicionar padrão à hierarquia por nível
            pattern_level = pattern.pattern_level.value
            self.pattern_hierarchy[f"level_{pattern_level}"].append(pattern.pattern_id)
            
        except Exception as e:
            logger.error(f"Erro na atualização da hierarquia de padrões: {e}")
    
    def validate_pattern_universality(self, pattern_id: str, 
                                   new_species_data: Dict[str, Any]) -> Dict[str, Any]:
        """Valida se um padrão é universal aplicando-o a uma nova espécie."""
        try:
            if pattern_id not in self.universal_patterns:
                return {"error": f"Padrão '{pattern_id}' não encontrado"}
            
            pattern = self.universal_patterns[pattern_id]
            
            # Extrair características da nova espécie
            new_characteristics = self._extract_characteristics(new_species_data)
            
            # Calcular similaridade com características do padrão
            similarity = self._calculate_pattern_similarity(pattern.characteristics, new_characteristics)
            
            # Determinar se o padrão se aplica
            applies = similarity >= 0.6  # Threshold de 60%
            
            # Atualizar confiança do padrão
            if applies:
                pattern.confidence = min(pattern.confidence + 0.1, 1.0)
                pattern.applicable_species.append(new_species_data.get("species", "unknown"))
                pattern.evidence_count += 1
            else:
                pattern.confidence = max(pattern.confidence - 0.05, 0.0)
            
            # Atualizar nível de confiança
            pattern.confidence_level = self._determine_confidence_level(pattern.confidence)
            pattern.last_updated = time.time()
            
            # Adicionar ao histórico de aprendizado
            pattern.learning_history.append({
                "species": new_species_data.get("species", "unknown"),
                "method": "validation",
                "confidence_gained": pattern.confidence,
                "similarity": similarity,
                "applies": applies,
                "timestamp": time.time()
            })
            
            # Salvar dados
            self._save_data()
            
            return {
                "pattern_id": pattern_id,
                "pattern_name": pattern.name,
                "applies": applies,
                "similarity": similarity,
                "updated_confidence": pattern.confidence,
                "confidence_level": pattern.confidence_level.value,
                "evidence_count": pattern.evidence_count,
                "applicable_species": pattern.applicable_species
            }
            
        except Exception as e:
            logger.error(f"Erro na validação de universalidade: {e}")
            return {"error": str(e)}
    
    def _calculate_pattern_similarity(self, pattern_chars: List[str], 
                                    new_chars: List[str]) -> float:
        """Calcula similaridade entre características de padrão e novas características."""
        try:
            if not pattern_chars or not new_chars:
                return 0.0
            
            # Converter para sets para cálculo de similaridade
            pattern_set = set(pattern_chars)
            new_set = set(new_chars)
            
            # Calcular similaridade de Jaccard
            intersection = len(pattern_set & new_set)
            union = len(pattern_set | new_set)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Erro no cálculo de similaridade de padrão: {e}")
            return 0.0
    
    def find_applicable_patterns(self, detection_data: Dict[str, Any], 
                               species: str = None) -> Dict[str, Any]:
        """Encontra padrões universais aplicáveis aos dados de detecção."""
        try:
            # Extrair características dos dados
            characteristics = self._extract_characteristics(detection_data)
            
            applicable_patterns = []
            
            # Verificar cada padrão universal
            for pattern_id, pattern in self.universal_patterns.items():
                # Calcular similaridade
                similarity = self._calculate_pattern_similarity(pattern.characteristics, characteristics)
                
                # Se similaridade for alta o suficiente
                if similarity >= 0.5:
                    applicable_patterns.append({
                        "pattern_id": pattern_id,
                        "pattern_name": pattern.name,
                        "pattern_type": pattern.pattern_type.value,
                        "pattern_level": pattern.pattern_level.value,
                        "similarity": similarity,
                        "confidence": pattern.confidence,
                        "confidence_level": pattern.confidence_level.value,
                        "characteristics": pattern.characteristics,
                        "applicable_species": pattern.applicable_species
                    })
            
            # Ordenar por similaridade
            applicable_patterns.sort(key=lambda x: x["similarity"], reverse=True)
            
            return {
                "detection_data": detection_data,
                "species": species,
                "extracted_characteristics": characteristics,
                "applicable_patterns": applicable_patterns,
                "total_applicable": len(applicable_patterns),
                "best_match": applicable_patterns[0] if applicable_patterns else None
            }
            
        except Exception as e:
            logger.error(f"Erro na busca de padrões aplicáveis: {e}")
            return {"error": str(e)}
    
    def get_universal_pattern_analysis(self, pattern_id: str = None) -> Dict[str, Any]:
        """Obtém análise de padrões universais."""
        try:
            if pattern_id:
                # Análise específica de um padrão
                if pattern_id not in self.universal_patterns:
                    return {"error": f"Padrão '{pattern_id}' não encontrado"}
                
                pattern = self.universal_patterns[pattern_id]
                
                return {
                    "pattern": asdict(pattern),
                    "learning_history": pattern.learning_history,
                    "validation_score": pattern.validation_score,
                    "universality_score": self._calculate_universality_score(pattern)
                }
            else:
                # Análise geral
                return {
                    "total_patterns": len(self.universal_patterns),
                    "total_learning_results": len(self.learning_results),
                    "pattern_distribution": self._get_pattern_distribution(),
                    "confidence_distribution": self._get_confidence_distribution(),
                    "pattern_hierarchy": dict(self.pattern_hierarchy),
                    "most_common_patterns": self._get_most_common_patterns(),
                    "universality_statistics": self._get_universality_statistics()
                }
                
        except Exception as e:
            logger.error(f"Erro na análise de padrões universais: {e}")
            return {"error": str(e)}
    
    def _calculate_universality_score(self, pattern: UniversalPattern) -> float:
        """Calcula score de universalidade de um padrão."""
        try:
            # Baseado no número de espécies aplicáveis e confiança
            species_count = len(pattern.applicable_species)
            confidence_factor = pattern.confidence
            evidence_factor = min(pattern.evidence_count / 10.0, 1.0)
            
            universality_score = (species_count * 0.4 + 
                                confidence_factor * 0.4 + 
                                evidence_factor * 0.2)
            
            return min(universality_score, 1.0)
            
        except Exception as e:
            logger.error(f"Erro no cálculo de score de universalidade: {e}")
            return 0.0
    
    def _get_pattern_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de padrões por tipo."""
        distribution = defaultdict(int)
        
        for pattern in self.universal_patterns.values():
            distribution[pattern.pattern_type.value] += 1
        
        return dict(distribution)
    
    def _get_confidence_distribution(self) -> Dict[str, int]:
        """Obtém distribuição de padrões por nível de confiança."""
        distribution = defaultdict(int)
        
        for pattern in self.universal_patterns.values():
            distribution[pattern.confidence_level.value] += 1
        
        return dict(distribution)
    
    def _get_most_common_patterns(self) -> List[Dict[str, Any]]:
        """Obtém padrões mais comuns."""
        patterns_with_scores = []
        
        for pattern in self.universal_patterns.values():
            universality_score = self._calculate_universality_score(pattern)
            patterns_with_scores.append({
                "pattern_id": pattern.pattern_id,
                "pattern_name": pattern.name,
                "pattern_type": pattern.pattern_type.value,
                "universality_score": universality_score,
                "confidence": pattern.confidence,
                "applicable_species_count": len(pattern.applicable_species)
            })
        
        # Ordenar por score de universalidade
        patterns_with_scores.sort(key=lambda x: x["universality_score"], reverse=True)
        
        return patterns_with_scores[:5]  # Top 5
    
    def _get_universality_statistics(self) -> Dict[str, Any]:
        """Obtém estatísticas de universalidade."""
        try:
            total_patterns = len(self.universal_patterns)
            if total_patterns == 0:
                return {"total_patterns": 0, "average_universality": 0.0}
            
            universality_scores = []
            for pattern in self.universal_patterns.values():
                score = self._calculate_universality_score(pattern)
                universality_scores.append(score)
            
            return {
                "total_patterns": total_patterns,
                "average_universality": sum(universality_scores) / len(universality_scores),
                "max_universality": max(universality_scores),
                "min_universality": min(universality_scores),
                "high_universality_count": len([s for s in universality_scores if s >= 0.7])
            }
            
        except Exception as e:
            logger.error(f"Erro nas estatísticas de universalidade: {e}")
            return {"error": str(e)}
    
    def _initialize_basic_patterns(self):
        """Inicializa padrões básicos universais."""
        # Padrões básicos para pássaros
        basic_patterns = [
            {
                "name": "Padrão Morfológico de Aves",
                "pattern_type": PatternType.MORPHOLOGICAL,
                "pattern_level": PatternLevel.CLASS_LEVEL,
                "characteristics": ["has_feathers", "has_beak", "has_wings", "has_tail"],
                "species": ["bird", "passaro"]
            },
            {
                "name": "Padrão Comportamental de Voo",
                "pattern_type": PatternType.BEHAVIORAL,
                "pattern_level": PatternLevel.ORDER_LEVEL,
                "characteristics": ["can_fly", "uses_wings", "aerial_movement"],
                "species": ["bird", "passaro"]
            }
        ]
        
        for pattern_data in basic_patterns:
            try:
                pattern_id = f"basic_{int(time.time())}_{random.randint(1000, 9999)}"
                pattern = UniversalPattern(
                    pattern_id=pattern_id,
                    name=pattern_data["name"],
                    pattern_type=pattern_data["pattern_type"],
                    pattern_level=pattern_data["pattern_level"],
                    description=f"Padrão básico {pattern_data['pattern_type'].value}",
                    characteristics=pattern_data["characteristics"],
                    applicable_species=pattern_data["species"],
                    confidence=0.8,
                    confidence_level=PatternConfidence.HIGH,
                    evidence_count=1,
                    validation_score=0.8,
                    learning_history=[{
                        "species": "basic",
                        "method": "initialization",
                        "confidence_gained": 0.8,
                        "timestamp": time.time()
                    }],
                    created_at=time.time(),
                    last_updated=time.time()
                )
                
                self.universal_patterns[pattern_id] = pattern
                
                for species in pattern_data["species"]:
                    self.species_patterns[species].append(pattern_id)
                
                self._update_pattern_hierarchy(pattern)
                
            except Exception as e:
                logger.error(f"Erro na inicialização de padrão básico: {e}")
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/universal_pattern_learning.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar padrões universais
                if "universal_patterns" in data:
                    for pattern_id, pattern_data in data["universal_patterns"].items():
                        pattern_data["pattern_type"] = PatternType(pattern_data["pattern_type"])
                        pattern_data["pattern_level"] = PatternLevel(pattern_data["pattern_level"])
                        pattern_data["confidence_level"] = PatternConfidence(pattern_data["confidence_level"])
                        self.universal_patterns[pattern_id] = UniversalPattern(**pattern_data)
                
                # Carregar resultados de aprendizado
                if "learning_results" in data:
                    for result_data in data["learning_results"]:
                        self.learning_results.append(PatternLearningResult(**result_data))
                
                # Carregar padrões por espécie
                if "species_patterns" in data:
                    self.species_patterns = defaultdict(list, data["species_patterns"])
                
                # Carregar hierarquia de padrões
                if "pattern_hierarchy" in data:
                    self.pattern_hierarchy = defaultdict(list, data["pattern_hierarchy"])
                
                logger.info(f"Dados de aprendizado por padrões universais carregados: {len(self.universal_patterns)} padrões")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de aprendizado por padrões universais: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/universal_pattern_learning.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "universal_patterns": {
                    pattern_id: {
                        "pattern_id": pattern.pattern_id,
                        "name": pattern.name,
                        "pattern_type": pattern.pattern_type.value,
                        "pattern_level": pattern.pattern_level.value,
                        "description": pattern.description,
                        "characteristics": pattern.characteristics,
                        "applicable_species": pattern.applicable_species,
                        "confidence": pattern.confidence,
                        "confidence_level": pattern.confidence_level.value,
                        "evidence_count": pattern.evidence_count,
                        "validation_score": pattern.validation_score,
                        "learning_history": pattern.learning_history,
                        "created_at": pattern.created_at,
                        "last_updated": pattern.last_updated
                    }
                    for pattern_id, pattern in self.universal_patterns.items()
                },
                "learning_results": [asdict(result) for result in self.learning_results],
                "species_patterns": dict(self.species_patterns),
                "pattern_hierarchy": dict(self.pattern_hierarchy),
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de aprendizado por padrões universais salvos: {len(self.universal_patterns)} padrões")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de aprendizado por padrões universais: {e}")
