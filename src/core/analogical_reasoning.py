#!/usr/bin/env python3
"""
Sistema de Raciocínio por Analogia
=================================

Este módulo implementa um sistema avançado de raciocínio por analogia que permite:
- Identificação de analogias estruturais entre conceitos
- Mapeamento de correspondências entre domínios
- Transferência de conhecimento baseada em analogias
- Raciocínio por similaridade funcional
- Sistema de analogias hierárquicas
- Analogias causais e temporais
- Sistema de validação de analogias
- Raciocínio por casos similares
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

class AnalogyType(Enum):
    """Tipos de analogia."""
    STRUCTURAL = "structural"           # Estrutural (baseada em estrutura)
    FUNCTIONAL = "functional"           # Funcional (baseada em função)
    CAUSAL = "causal"                  # Causal (baseada em causalidade)
    TEMPORAL = "temporal"              # Temporal (baseada em sequência temporal)
    SPATIAL = "spatial"                # Espacial (baseada em posição)
    BEHAVIORAL = "behavioral"          # Comportamental (baseada em comportamento)

class AnalogyStrength(Enum):
    """Força da analogia."""
    WEAK = "weak"                      # Fraca (0.0 - 0.3)
    MODERATE = "moderate"              # Moderada (0.3 - 0.6)
    STRONG = "strong"                  # Forte (0.6 - 0.8)
    VERY_STRONG = "very_strong"        # Muito forte (0.8 - 1.0)

class MappingType(Enum):
    """Tipos de mapeamento."""
    ONE_TO_ONE = "one_to_one"          # Um para um
    ONE_TO_MANY = "one_to_many"        # Um para muitos
    MANY_TO_ONE = "many_to_one"        # Muitos para um
    MANY_TO_MANY = "many_to_many"      # Muitos para muitos

@dataclass
class AnalogyMapping:
    """Mapeamento entre elementos de uma analogia."""
    source_element: str
    target_element: str
    mapping_type: MappingType
    confidence: float
    evidence: List[str]
    created_at: float

@dataclass
class Analogy:
    """Representa uma analogia."""
    analogy_id: str
    source_domain: str
    target_domain: str
    analogy_type: AnalogyType
    strength: AnalogyStrength
    mappings: List[AnalogyMapping]
    confidence: float
    evidence: List[str]
    usage_count: int
    success_rate: float
    created_at: float
    last_updated: float

@dataclass
class AnalogyResult:
    """Resultado de raciocínio por analogia."""
    result_id: str
    source_concept: str
    target_concept: str
    analogy_used: str
    inferred_properties: List[str]
    confidence: float
    reasoning_steps: List[str]
    supporting_evidence: List[str]
    timestamp: float

class StructuralAnalogyEngine:
    """Motor de analogia estrutural."""
    
    def __init__(self):
        self.analogies: Dict[str, Analogy] = {}
        self.analogy_results: List[AnalogyResult] = []
        self.concept_structures: Dict[str, Dict[str, Any]] = {}
        self.structural_patterns: Dict[str, List[str]] = defaultdict(list)
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar analogias básicas
        self._initialize_basic_analogies()
    
    def find_structural_analogy(self, source_concept: str, target_concept: str) -> Dict[str, Any]:
        """Encontra analogia estrutural entre dois conceitos."""
        try:
            if source_concept not in self.concept_structures:
                return {"error": f"Estrutura do conceito '{source_concept}' não encontrada"}
            
            if target_concept not in self.concept_structures:
                return {"error": f"Estrutura do conceito '{target_concept}' não encontrada"}
            
            source_structure = self.concept_structures[source_concept]
            target_structure = self.concept_structures[target_concept]
            
            # Calcular similaridade estrutural
            structural_similarity = self._calculate_structural_similarity(source_structure, target_structure)
            
            # Encontrar mapeamentos estruturais
            mappings = self._find_structural_mappings(source_structure, target_structure)
            
            # Calcular força da analogia
            analogy_strength = self._calculate_analogy_strength(structural_similarity, mappings)
            
            # Gerar evidências
            evidence = self._generate_structural_evidence(source_structure, target_structure, mappings)
            
            return {
                "source_concept": source_concept,
                "target_concept": target_concept,
                "analogy_type": AnalogyType.STRUCTURAL.value,
                "structural_similarity": structural_similarity,
                "analogy_strength": analogy_strength.value,
                "mappings": mappings,
                "evidence": evidence,
                "confidence": self._calculate_confidence(structural_similarity, mappings)
            }
            
        except Exception as e:
            logger.error(f"Erro na analogia estrutural: {e}")
            return {"error": str(e)}
    
    def _calculate_structural_similarity(self, source_structure: Dict[str, Any], 
                                      target_structure: Dict[str, Any]) -> float:
        """Calcula similaridade estrutural entre duas estruturas."""
        try:
            # Comparar componentes estruturais
            source_components = set(source_structure.get("components", []))
            target_components = set(target_structure.get("components", []))
            
            # Comparar relações estruturais
            source_relations = set(source_structure.get("relations", []))
            target_relations = set(target_structure.get("relations", []))
            
            # Comparar propriedades estruturais
            source_properties = set(source_structure.get("properties", []))
            target_properties = set(target_structure.get("properties", []))
            
            # Calcular similaridade de componentes
            component_similarity = len(source_components & target_components) / len(source_components | target_components) if source_components | target_components else 0
            
            # Calcular similaridade de relações
            relation_similarity = len(source_relations & target_relations) / len(source_relations | target_relations) if source_relations | target_relations else 0
            
            # Calcular similaridade de propriedades
            property_similarity = len(source_properties & target_properties) / len(source_properties | target_properties) if source_properties | target_properties else 0
            
            # Calcular similaridade estrutural geral
            structural_similarity = (component_similarity * 0.4 + 
                                   relation_similarity * 0.4 + 
                                   property_similarity * 0.2)
            
            return structural_similarity
            
        except Exception as e:
            logger.error(f"Erro no cálculo de similaridade estrutural: {e}")
            return 0.0
    
    def _find_structural_mappings(self, source_structure: Dict[str, Any], 
                                target_structure: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Encontra mapeamentos estruturais entre duas estruturas."""
        mappings = []
        
        try:
            # Mapear componentes
            source_components = source_structure.get("components", [])
            target_components = target_structure.get("components", [])
            
            for source_comp in source_components:
                for target_comp in target_components:
                    if self._are_components_similar(source_comp, target_comp):
                        mapping = {
                            "source_element": source_comp,
                            "target_element": target_comp,
                            "mapping_type": MappingType.ONE_TO_ONE.value,
                            "confidence": self._calculate_component_similarity(source_comp, target_comp),
                            "evidence": [f"Componente similar: {source_comp} -> {target_comp}"]
                        }
                        mappings.append(mapping)
            
            # Mapear relações
            source_relations = source_structure.get("relations", [])
            target_relations = target_structure.get("relations", [])
            
            for source_rel in source_relations:
                for target_rel in target_relations:
                    if self._are_relations_similar(source_rel, target_rel):
                        mapping = {
                            "source_element": source_rel,
                            "target_element": target_rel,
                            "mapping_type": MappingType.ONE_TO_ONE.value,
                            "confidence": self._calculate_relation_similarity(source_rel, target_rel),
                            "evidence": [f"Relação similar: {source_rel} -> {target_rel}"]
                        }
                        mappings.append(mapping)
            
            # Mapear propriedades
            source_properties = source_structure.get("properties", [])
            target_properties = target_structure.get("properties", [])
            
            for source_prop in source_properties:
                for target_prop in target_properties:
                    if self._are_properties_similar(source_prop, target_prop):
                        mapping = {
                            "source_element": source_prop,
                            "target_element": target_prop,
                            "mapping_type": MappingType.ONE_TO_ONE.value,
                            "confidence": self._calculate_property_similarity(source_prop, target_prop),
                            "evidence": [f"Propriedade similar: {source_prop} -> {target_prop}"]
                        }
                        mappings.append(mapping)
            
            return mappings
            
        except Exception as e:
            logger.error(f"Erro no mapeamento estrutural: {e}")
            return []
    
    def _are_components_similar(self, comp1: str, comp2: str) -> bool:
        """Verifica se dois componentes são similares."""
        # Implementação simples baseada em similaridade de string
        similarity = self._calculate_string_similarity(comp1, comp2)
        return similarity > 0.6
    
    def _are_relations_similar(self, rel1: str, rel2: str) -> bool:
        """Verifica se duas relações são similares."""
        similarity = self._calculate_string_similarity(rel1, rel2)
        return similarity > 0.7
    
    def _are_properties_similar(self, prop1: str, prop2: str) -> bool:
        """Verifica se duas propriedades são similares."""
        similarity = self._calculate_string_similarity(prop1, prop2)
        return similarity > 0.8
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calcula similaridade entre duas strings."""
        try:
            # Implementação simples de similaridade de Jaccard
            set1 = set(str1.lower().split())
            set2 = set(str2.lower().split())
            
            intersection = len(set1 & set2)
            union = len(set1 | set2)
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Erro no cálculo de similaridade de string: {e}")
            return 0.0
    
    def _calculate_component_similarity(self, comp1: str, comp2: str) -> float:
        """Calcula similaridade entre componentes."""
        return self._calculate_string_similarity(comp1, comp2)
    
    def _calculate_relation_similarity(self, rel1: str, rel2: str) -> float:
        """Calcula similaridade entre relações."""
        return self._calculate_string_similarity(rel1, rel2)
    
    def _calculate_property_similarity(self, prop1: str, prop2: str) -> float:
        """Calcula similaridade entre propriedades."""
        return self._calculate_string_similarity(prop1, prop2)
    
    def _calculate_analogy_strength(self, structural_similarity: float, 
                                  mappings: List[Dict[str, Any]]) -> AnalogyStrength:
        """Calcula força da analogia."""
        if structural_similarity >= 0.8 and len(mappings) >= 3:
            return AnalogyStrength.VERY_STRONG
        elif structural_similarity >= 0.6 and len(mappings) >= 2:
            return AnalogyStrength.STRONG
        elif structural_similarity >= 0.3 and len(mappings) >= 1:
            return AnalogyStrength.MODERATE
        else:
            return AnalogyStrength.WEAK
    
    def _generate_structural_evidence(self, source_structure: Dict[str, Any], 
                                    target_structure: Dict[str, Any], 
                                    mappings: List[Dict[str, Any]]) -> List[str]:
        """Gera evidências para analogia estrutural."""
        evidence = []
        
        try:
            # Evidências baseadas em componentes
            source_components = source_structure.get("components", [])
            target_components = target_structure.get("components", [])
            
            common_components = set(source_components) & set(target_components)
            if common_components:
                evidence.append(f"Componentes comuns: {', '.join(common_components)}")
            
            # Evidências baseadas em relações
            source_relations = source_structure.get("relations", [])
            target_relations = target_structure.get("relations", [])
            
            common_relations = set(source_relations) & set(target_relations)
            if common_relations:
                evidence.append(f"Relações comuns: {', '.join(common_relations)}")
            
            # Evidências baseadas em propriedades
            source_properties = source_structure.get("properties", [])
            target_properties = target_structure.get("properties", [])
            
            common_properties = set(source_properties) & set(target_properties)
            if common_properties:
                evidence.append(f"Propriedades comuns: {', '.join(common_properties)}")
            
            # Evidências baseadas em mapeamentos
            if mappings:
                evidence.append(f"Mapeamentos estruturais encontrados: {len(mappings)}")
            
            return evidence
            
        except Exception as e:
            logger.error(f"Erro na geração de evidências: {e}")
            return []
    
    def _calculate_confidence(self, structural_similarity: float, 
                           mappings: List[Dict[str, Any]]) -> float:
        """Calcula confiança na analogia."""
        if not mappings:
            return 0.0
        
        # Calcular confiança média dos mapeamentos
        mapping_confidences = [mapping.get("confidence", 0.0) for mapping in mappings]
        average_mapping_confidence = sum(mapping_confidences) / len(mapping_confidences)
        
        # Combinar similaridade estrutural e confiança dos mapeamentos
        confidence = (structural_similarity * 0.6 + average_mapping_confidence * 0.4)
        
        return min(confidence, 1.0)
    
    def add_concept_structure(self, concept_name: str, structure: Dict[str, Any]) -> bool:
        """Adiciona estrutura de um conceito."""
        try:
            self.concept_structures[concept_name] = structure
            
            # Atualizar padrões estruturais
            self._update_structural_patterns(concept_name, structure)
            
            # Salvar dados
            self._save_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar estrutura de conceito: {e}")
            return False
    
    def _update_structural_patterns(self, concept_name: str, structure: Dict[str, Any]):
        """Atualiza padrões estruturais."""
        try:
            # Extrair padrões da estrutura
            components = structure.get("components", [])
            relations = structure.get("relations", [])
            properties = structure.get("properties", [])
            
            # Adicionar aos padrões estruturais
            for component in components:
                self.structural_patterns["components"].append(component)
            
            for relation in relations:
                self.structural_patterns["relations"].append(relation)
            
            for property_name in properties:
                self.structural_patterns["properties"].append(property_name)
            
        except Exception as e:
            logger.error(f"Erro na atualização de padrões estruturais: {e}")
    
    def reason_by_analogy(self, source_concept: str, target_concept: str, 
                         property_to_transfer: str = None) -> Dict[str, Any]:
        """Raciocina por analogia entre dois conceitos."""
        try:
            # Encontrar analogia estrutural
            analogy_result = self.find_structural_analogy(source_concept, target_concept)
            
            if "error" in analogy_result:
                return analogy_result
            
            # Transferir propriedades baseado na analogia
            transferred_properties = self._transfer_properties_by_analogy(
                source_concept, target_concept, analogy_result, property_to_transfer
            )
            
            # Gerar raciocínio
            reasoning_steps = self._generate_analogy_reasoning(
                source_concept, target_concept, analogy_result, transferred_properties
            )
            
            # Criar resultado
            result = AnalogyResult(
                result_id=f"analogy_{int(time.time())}",
                source_concept=source_concept,
                target_concept=target_concept,
                analogy_used=analogy_result["analogy_type"],
                inferred_properties=transferred_properties,
                confidence=analogy_result["confidence"],
                reasoning_steps=reasoning_steps,
                supporting_evidence=analogy_result["evidence"],
                timestamp=time.time()
            )
            
            # Adicionar ao histórico
            self.analogy_results.append(result)
            
            return {
                "source_concept": source_concept,
                "target_concept": target_concept,
                "analogy_result": analogy_result,
                "transferred_properties": transferred_properties,
                "reasoning_steps": reasoning_steps,
                "confidence": analogy_result["confidence"],
                "analogy_result": asdict(result)
            }
            
        except Exception as e:
            logger.error(f"Erro no raciocínio por analogia: {e}")
            return {"error": str(e)}
    
    def _transfer_properties_by_analogy(self, source_concept: str, target_concept: str, 
                                     analogy_result: Dict[str, Any], 
                                     property_to_transfer: str = None) -> List[str]:
        """Transfere propriedades baseado na analogia."""
        transferred_properties = []
        
        try:
            source_structure = self.concept_structures.get(source_concept, {})
            source_properties = source_structure.get("properties", [])
            
            if property_to_transfer:
                # Transferir propriedade específica
                if property_to_transfer in source_properties:
                    transferred_properties.append(property_to_transfer)
            else:
                # Transferir propriedades baseado na força da analogia
                analogy_strength = analogy_result.get("analogy_strength", "weak")
                
                if analogy_strength in ["strong", "very_strong"]:
                    # Transferir todas as propriedades
                    transferred_properties.extend(source_properties)
                elif analogy_strength == "moderate":
                    # Transferir propriedades mais relevantes
                    transferred_properties.extend(source_properties[:3])
                else:
                    # Transferir apenas propriedades básicas
                    transferred_properties.extend(source_properties[:1])
            
            return transferred_properties
            
        except Exception as e:
            logger.error(f"Erro na transferência de propriedades: {e}")
            return []
    
    def _generate_analogy_reasoning(self, source_concept: str, target_concept: str, 
                                  analogy_result: Dict[str, Any], 
                                  transferred_properties: List[str]) -> List[str]:
        """Gera raciocínio baseado na analogia."""
        reasoning_steps = []
        
        try:
            reasoning_steps.append(f"Analisando analogia entre '{source_concept}' e '{target_concept}'")
            
            analogy_strength = analogy_result.get("analogy_strength", "weak")
            reasoning_steps.append(f"Força da analogia: {analogy_strength}")
            
            structural_similarity = analogy_result.get("structural_similarity", 0.0)
            reasoning_steps.append(f"Similaridade estrutural: {structural_similarity:.3f}")
            
            mappings = analogy_result.get("mappings", [])
            reasoning_steps.append(f"Mapeamentos encontrados: {len(mappings)}")
            
            if transferred_properties:
                reasoning_steps.append(f"Propriedades transferidas: {', '.join(transferred_properties)}")
            
            return reasoning_steps
            
        except Exception as e:
            logger.error(f"Erro na geração de raciocínio: {e}")
            return []
    
    def get_analogy_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """Obtém análise de analogias."""
        try:
            if concept_name:
                # Análise específica de um conceito
                if concept_name not in self.concept_structures:
                    return {"error": f"Estrutura do conceito '{concept_name}' não encontrada"}
                
                structure = self.concept_structures[concept_name]
                
                # Encontrar analogias relacionadas
                related_analogies = []
                for other_concept in self.concept_structures:
                    if other_concept != concept_name:
                        analogy_result = self.find_structural_analogy(concept_name, other_concept)
                        if analogy_result.get("confidence", 0) > 0.3:
                            related_analogies.append({
                                "target_concept": other_concept,
                                "confidence": analogy_result.get("confidence", 0),
                                "strength": analogy_result.get("analogy_strength", "weak")
                            })
                
                return {
                    "concept": concept_name,
                    "structure": structure,
                    "related_analogies": related_analogies,
                    "total_analogies": len(related_analogies)
                }
            else:
                # Análise geral
                return {
                    "total_concepts": len(self.concept_structures),
                    "total_analogies": len(self.analogies),
                    "total_results": len(self.analogy_results),
                    "structural_patterns": {
                        "components": len(self.structural_patterns["components"]),
                        "relations": len(self.structural_patterns["relations"]),
                        "properties": len(self.structural_patterns["properties"])
                    },
                    "most_common_patterns": self._get_most_common_patterns()
                }
                
        except Exception as e:
            logger.error(f"Erro na análise de analogias: {e}")
            return {"error": str(e)}
    
    def _get_most_common_patterns(self) -> Dict[str, List[Tuple[str, int]]]:
        """Obtém padrões mais comuns."""
        patterns = {}
        
        for pattern_type, pattern_list in self.structural_patterns.items():
            pattern_counts = Counter(pattern_list)
            patterns[pattern_type] = pattern_counts.most_common(5)
        
        return patterns
    
    def _initialize_basic_analogies(self):
        """Inicializa analogias básicas."""
        # Adicionar estruturas básicas de conceitos comuns
        basic_structures = {
            "bird": {
                "components": ["head", "body", "wings", "tail", "legs"],
                "relations": ["head_connected_to_body", "wings_attached_to_body", "tail_extends_from_body"],
                "properties": ["has_feathers", "can_fly", "has_beak", "lays_eggs"]
            },
            "airplane": {
                "components": ["cockpit", "fuselage", "wings", "tail", "landing_gear"],
                "relations": ["cockpit_connected_to_fuselage", "wings_attached_to_fuselage", "tail_extends_from_fuselage"],
                "properties": ["can_fly", "has_engine", "transports_passengers", "has_wings"]
            },
            "car": {
                "components": ["engine", "body", "wheels", "doors", "windshield"],
                "relations": ["engine_inside_body", "wheels_attached_to_body", "doors_attached_to_body"],
                "properties": ["has_engine", "transports_passengers", "has_wheels", "has_doors"]
            }
        }
        
        for concept_name, structure in basic_structures.items():
            self.add_concept_structure(concept_name, structure)
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/analogical_reasoning.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar estruturas de conceitos
                if "concept_structures" in data:
                    self.concept_structures = data["concept_structures"]
                
                # Carregar analogias
                if "analogies" in data:
                    for analogy_id, analogy_data in data["analogies"].items():
                        analogy_data["analogy_type"] = AnalogyType(analogy_data["analogy_type"])
                        analogy_data["strength"] = AnalogyStrength(analogy_data["strength"])
                        self.analogies[analogy_id] = Analogy(**analogy_data)
                
                # Carregar resultados de analogia
                if "analogy_results" in data:
                    for result_data in data["analogy_results"]:
                        self.analogy_results.append(AnalogyResult(**result_data))
                
                logger.info(f"Dados de raciocínio por analogia carregados: {len(self.concept_structures)} conceitos, {len(self.analogies)} analogias")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de raciocínio por analogia: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/analogical_reasoning.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "concept_structures": self.concept_structures,
                "analogies": {
                    analogy_id: {
                        "analogy_id": analogy.analogy_id,
                        "source_domain": analogy.source_domain,
                        "target_domain": analogy.target_domain,
                        "analogy_type": analogy.analogy_type.value,
                        "strength": analogy.strength.value,
                        "mappings": [asdict(mapping) for mapping in analogy.mappings],
                        "confidence": analogy.confidence,
                        "evidence": analogy.evidence,
                        "usage_count": analogy.usage_count,
                        "success_rate": analogy.success_rate,
                        "created_at": analogy.created_at,
                        "last_updated": analogy.last_updated
                    }
                    for analogy_id, analogy in self.analogies.items()
                },
                "analogy_results": [asdict(result) for result in self.analogy_results],
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de raciocínio por analogia salvos: {len(self.concept_structures)} conceitos, {len(self.analogies)} analogias")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de raciocínio por analogia: {e}")
