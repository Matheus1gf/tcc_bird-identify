#!/usr/bin/env python3
"""
Sistema de Raciocínio Conceitual
===============================

Este módulo implementa um sistema avançado de raciocínio conceitual que permite:
- Inferência abstrata sobre conceitos e relações
- Raciocínio lógico sobre propriedades e características
- Dedução de propriedades baseada em conceitos conhecidos
- Abstração de características essenciais
- Raciocínio por analogia e similaridade
- Inferência causal entre conceitos
- Sistema de regras de inferência
- Raciocínio probabilístico sobre conceitos
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

class InferenceType(Enum):
    """Tipos de inferência."""
    DEDUCTIVE = "deductive"           # Dedutiva (geral para específico)
    INDUCTIVE = "inductive"           # Indutiva (específico para geral)
    ABDUCTIVE = "abductive"           # Abdutiva (explicação mais provável)
    ANALOGICAL = "analogical"         # Por analogia
    CAUSAL = "causal"                 # Causal
    PROBABILISTIC = "probabilistic"   # Probabilística

class ConceptRelation(Enum):
    """Tipos de relações entre conceitos."""
    IS_A = "is_a"                     # É um tipo de
    HAS_A = "has_a"                   # Tem uma propriedade
    PART_OF = "part_of"               # É parte de
    CAUSES = "causes"                 # Causa
    SIMILAR_TO = "similar_to"         # Similar a
    OPPOSITE_OF = "opposite_of"       # Oposto de
    REQUIRES = "requires"             # Requer
    ENABLES = "enables"               # Habilita

class AbstractionLevel(Enum):
    """Níveis de abstração."""
    CONCRETE = "concrete"             # Concreto (específico)
    SPECIFIC = "specific"            # Específico
    GENERAL = "general"              # Geral
    ABSTRACT = "abstract"             # Abstrato
    UNIVERSAL = "universal"          # Universal

@dataclass
class Concept:
    """Representa um conceito."""
    name: str
    definition: str
    properties: List[str]
    abstraction_level: AbstractionLevel
    confidence: float
    examples: List[str]
    relations: Dict[str, List[str]]
    created_at: float
    last_updated: float

@dataclass
class InferenceRule:
    """Regra de inferência."""
    rule_id: str
    premise_pattern: List[str]
    conclusion_pattern: List[str]
    inference_type: InferenceType
    confidence: float
    usage_count: int
    success_rate: float
    created_at: float
    last_updated: float

@dataclass
class InferenceResult:
    """Resultado de uma inferência."""
    inference_id: str
    premise_concepts: List[str]
    conclusion_concepts: List[str]
    inference_type: InferenceType
    confidence: float
    reasoning_steps: List[str]
    supporting_evidence: List[str]
    timestamp: float

class AbstractInferenceEngine:
    """Motor de inferência abstrata."""
    
    def __init__(self):
        self.concepts: Dict[str, Concept] = {}
        self.inference_rules: List[InferenceRule] = []
        self.inference_history: List[InferenceResult] = []
        self.concept_graph: Dict[str, Set[str]] = defaultdict(set)
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar regras básicas
        self._initialize_basic_rules()
    
    def infer_abstract_properties(self, concept_name: str, target_property: str = None) -> Dict[str, Any]:
        """Infere propriedades abstratas de um conceito."""
        try:
            if concept_name not in self.concepts:
                return {"error": f"Conceito '{concept_name}' não encontrado"}
            
            concept = self.concepts[concept_name]
            inferences = []
            
            # Inferência dedutiva
            deductive_results = self._deductive_inference(concept, target_property)
            inferences.extend(deductive_results)
            
            # Inferência indutiva
            inductive_results = self._inductive_inference(concept, target_property)
            inferences.extend(inductive_results)
            
            # Inferência abdutiva
            abductive_results = self._abductive_inference(concept, target_property)
            inferences.extend(abductive_results)
            
            # Inferência por analogia
            analogical_results = self._analogical_inference(concept, target_property)
            inferences.extend(analogical_results)
            
            # Inferência causal
            causal_results = self._causal_inference(concept, target_property)
            inferences.extend(causal_results)
            
            # Inferência probabilística
            probabilistic_results = self._probabilistic_inference(concept, target_property)
            inferences.extend(probabilistic_results)
            
            # Consolidar resultados
            consolidated_results = self._consolidate_inferences(inferences)
            
            return {
                "concept": concept_name,
                "target_property": target_property,
                "inferences": consolidated_results,
                "total_inferences": len(inferences),
                "confidence_score": self._calculate_confidence_score(consolidated_results),
                "reasoning_summary": self._generate_reasoning_summary(consolidated_results)
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência abstrata: {e}")
            return {"error": str(e)}
    
    def _deductive_inference(self, concept: Concept, target_property: str = None) -> List[Dict[str, Any]]:
        """Inferência dedutiva (geral para específico)."""
        inferences = []
        
        # Buscar conceitos mais gerais
        for relation_type, related_concepts in concept.relations.items():
            if relation_type == "is_a":
                for parent_concept_name in related_concepts:
                    if parent_concept_name in self.concepts:
                        parent_concept = self.concepts[parent_concept_name]
                        
                        # Herdar propriedades do conceito pai
                        for property_name in parent_concept.properties:
                            if target_property is None or target_property.lower() in property_name.lower():
                                inference = {
                                    "type": InferenceType.DEDUCTIVE.value,
                                    "property": property_name,
                                    "confidence": parent_concept.confidence * 0.9,
                                    "reasoning": f"'{concept.name}' herda '{property_name}' de '{parent_concept_name}'",
                                    "evidence": [f"Relação IS_A com {parent_concept_name}", f"Propriedade em {parent_concept_name}"]
                                }
                                inferences.append(inference)
        
        return inferences
    
    def _inductive_inference(self, concept: Concept, target_property: str = None) -> List[Dict[str, Any]]:
        """Inferência indutiva (específico para geral)."""
        inferences = []
        
        # Buscar conceitos similares
        similar_concepts = self._find_similar_concepts(concept)
        
        for similar_concept_name, similarity in similar_concepts:
            similar_concept = self.concepts[similar_concept_name]
            
            # Propriedades comuns
            common_properties = set(concept.properties) & set(similar_concept.properties)
            
            for property_name in common_properties:
                if target_property is None or target_property.lower() in property_name.lower():
                    inference = {
                        "type": InferenceType.INDUCTIVE.value,
                        "property": property_name,
                        "confidence": similarity * 0.8,
                        "reasoning": f"'{concept.name}' e '{similar_concept_name}' compartilham '{property_name}'",
                        "evidence": [f"Similaridade: {similarity:.3f}", f"Propriedade comum: {property_name}"]
                    }
                    inferences.append(inference)
        
        return inferences
    
    def _abductive_inference(self, concept: Concept, target_property: str = None) -> List[Dict[str, Any]]:
        """Inferência abdutiva (explicação mais provável)."""
        inferences = []
        
        # Buscar explicações para propriedades conhecidas
        for property_name in concept.properties:
            if target_property is None or target_property.lower() in property_name.lower():
                # Buscar conceitos que poderiam explicar essa propriedade
                explanatory_concepts = self._find_explanatory_concepts(property_name)
                
                for explanatory_concept_name, explanation_strength in explanatory_concepts:
                    inference = {
                        "type": InferenceType.ABDUCTIVE.value,
                        "property": property_name,
                        "confidence": explanation_strength * 0.7,
                        "reasoning": f"'{explanatory_concept_name}' explica '{property_name}' em '{concept.name}'",
                        "evidence": [f"Explicação: {explanatory_concept_name}", f"Força: {explanation_strength:.3f}"]
                    }
                    inferences.append(inference)
        
        return inferences
    
    def _analogical_inference(self, concept: Concept, target_property: str = None) -> List[Dict[str, Any]]:
        """Inferência por analogia."""
        inferences = []
        
        # Buscar analogias baseadas em estrutura
        analogies = self._find_structural_analogies(concept)
        
        for analogy_concept_name, analogy_strength in analogies:
            analogy_concept = self.concepts[analogy_concept_name]
            
            # Propriedades que poderiam ser transferidas por analogia
            for property_name in analogy_concept.properties:
                if target_property is None or target_property.lower() in property_name.lower():
                    inference = {
                        "type": InferenceType.ANALOGICAL.value,
                        "property": property_name,
                        "confidence": analogy_strength * 0.6,
                        "reasoning": f"Analogia entre '{concept.name}' e '{analogy_concept_name}' sugere '{property_name}'",
                        "evidence": [f"Analogia: {analogy_strength:.3f}", f"Propriedade em {analogy_concept_name}"]
                    }
                    inferences.append(inference)
        
        return inferences
    
    def _causal_inference(self, concept: Concept, target_property: str = None) -> List[Dict[str, Any]]:
        """Inferência causal."""
        inferences = []
        
        # Buscar relações causais
        for relation_type, related_concepts in concept.relations.items():
            if relation_type in ["causes", "enables", "requires"]:
                for related_concept_name in related_concepts:
                    if related_concept_name in self.concepts:
                        related_concept = self.concepts[related_concept_name]
                        
                        # Propriedades que poderiam ser causadas
                        for property_name in related_concept.properties:
                            if target_property is None or target_property.lower() in property_name.lower():
                                inference = {
                                    "type": InferenceType.CAUSAL.value,
                                    "property": property_name,
                                    "confidence": concept.confidence * 0.8,
                                    "reasoning": f"'{concept.name}' {relation_type} '{property_name}' via '{related_concept_name}'",
                                    "evidence": [f"Relação causal: {relation_type}", f"Conceito relacionado: {related_concept_name}"]
                                }
                                inferences.append(inference)
        
        return inferences
    
    def _probabilistic_inference(self, concept: Concept, target_property: str = None) -> List[Dict[str, Any]]:
        """Inferência probabilística."""
        inferences = []
        
        # Calcular probabilidades baseadas em frequência
        property_frequencies = self._calculate_property_frequencies()
        
        for property_name, frequency in property_frequencies.items():
            if target_property is None or target_property.lower() in property_name.lower():
                # Ajustar probabilidade baseada no nível de abstração
                abstraction_factor = self._get_abstraction_factor(concept.abstraction_level)
                probability = frequency * abstraction_factor
                
                inference = {
                    "type": InferenceType.PROBABILISTIC.value,
                    "property": property_name,
                    "confidence": probability,
                    "reasoning": f"Probabilidade de '{property_name}' baseada em frequência: {frequency:.3f}",
                    "evidence": [f"Frequência: {frequency:.3f}", f"Fator de abstração: {abstraction_factor:.3f}"]
                }
                inferences.append(inference)
        
        return inferences
    
    def _find_similar_concepts(self, concept: Concept) -> List[Tuple[str, float]]:
        """Encontra conceitos similares."""
        similarities = []
        
        for other_concept_name, other_concept in self.concepts.items():
            if other_concept_name == concept.name:
                continue
            
            # Calcular similaridade baseada em propriedades
            common_properties = set(concept.properties) & set(other_concept.properties)
            total_properties = set(concept.properties) | set(other_concept.properties)
            
            if total_properties:
                similarity = len(common_properties) / len(total_properties)
                similarities.append((other_concept_name, similarity))
        
        # Ordenar por similaridade
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:5]  # Top 5
    
    def _find_explanatory_concepts(self, property_name: str) -> List[Tuple[str, float]]:
        """Encontra conceitos que podem explicar uma propriedade."""
        explanations = []
        
        for concept_name, concept in self.concepts.items():
            if property_name in concept.properties:
                # Calcular força da explicação baseada na confiança e frequência
                explanation_strength = concept.confidence * 0.8
                explanations.append((concept_name, explanation_strength))
        
        # Ordenar por força da explicação
        explanations.sort(key=lambda x: x[1], reverse=True)
        return explanations[:3]  # Top 3
    
    def _find_structural_analogies(self, concept: Concept) -> List[Tuple[str, float]]:
        """Encontra analogias estruturais."""
        analogies = []
        
        for other_concept_name, other_concept in self.concepts.items():
            if other_concept_name == concept.name:
                continue
            
            # Calcular analogia baseada em estrutura de relações
            common_relations = 0
            total_relations = 0
            
            for relation_type in set(concept.relations.keys()) | set(other_concept.relations.keys()):
                concept_relations = set(concept.relations.get(relation_type, []))
                other_relations = set(other_concept.relations.get(relation_type, []))
                
                common_relations += len(concept_relations & other_relations)
                total_relations += len(concept_relations | other_relations)
            
            if total_relations > 0:
                analogy_strength = common_relations / total_relations
                analogies.append((other_concept_name, analogy_strength))
        
        # Ordenar por força da analogia
        analogies.sort(key=lambda x: x[1], reverse=True)
        return analogies[:3]  # Top 3
    
    def _calculate_property_frequencies(self) -> Dict[str, float]:
        """Calcula frequências de propriedades."""
        property_counts = Counter()
        total_concepts = len(self.concepts)
        
        for concept in self.concepts.values():
            for property_name in concept.properties:
                property_counts[property_name] += 1
        
        # Converter para frequências
        frequencies = {}
        for property_name, count in property_counts.items():
            frequencies[property_name] = count / total_concepts if total_concepts > 0 else 0
        
        return frequencies
    
    def _get_abstraction_factor(self, abstraction_level: AbstractionLevel) -> float:
        """Obtém fator de abstração."""
        factors = {
            AbstractionLevel.CONCRETE: 0.3,
            AbstractionLevel.SPECIFIC: 0.5,
            AbstractionLevel.GENERAL: 0.7,
            AbstractionLevel.ABSTRACT: 0.9,
            AbstractionLevel.UNIVERSAL: 1.0
        }
        return factors.get(abstraction_level, 0.5)
    
    def _consolidate_inferences(self, inferences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Consolida inferências similares."""
        consolidated = {}
        
        for inference in inferences:
            property_name = inference["property"]
            
            if property_name not in consolidated:
                consolidated[property_name] = {
                    "property": property_name,
                    "inferences": [],
                    "max_confidence": 0.0,
                    "total_confidence": 0.0,
                    "inference_types": set()
                }
            
            consolidated[property_name]["inferences"].append(inference)
            consolidated[property_name]["max_confidence"] = max(
                consolidated[property_name]["max_confidence"], 
                inference["confidence"]
            )
            consolidated[property_name]["total_confidence"] += inference["confidence"]
            consolidated[property_name]["inference_types"].add(inference["type"])
        
        # Calcular confiança média
        for property_name in consolidated:
            count = len(consolidated[property_name]["inferences"])
            consolidated[property_name]["average_confidence"] = (
                consolidated[property_name]["total_confidence"] / count
            )
            consolidated[property_name]["inference_types"] = list(consolidated[property_name]["inference_types"])
        
        return consolidated
    
    def _calculate_confidence_score(self, consolidated_results: Dict[str, Any]) -> float:
        """Calcula score de confiança geral."""
        if not consolidated_results:
            return 0.0
        
        total_confidence = sum(
            result["average_confidence"] for result in consolidated_results.values()
        )
        return total_confidence / len(consolidated_results)
    
    def _generate_reasoning_summary(self, consolidated_results: Dict[str, Any]) -> List[str]:
        """Gera resumo do raciocínio."""
        summary = []
        
        for property_name, result in consolidated_results.items():
            inference_types = result["inference_types"]
            confidence = result["average_confidence"]
            
            summary.append(
                f"'{property_name}': {confidence:.3f} confiança via {', '.join(inference_types)}"
            )
        
        return summary
    
    def add_concept(self, name: str, definition: str, properties: List[str], 
                   abstraction_level: AbstractionLevel = AbstractionLevel.GENERAL,
                   relations: Dict[str, List[str]] = None) -> bool:
        """Adiciona um conceito."""
        try:
            concept = Concept(
                name=name,
                definition=definition,
                properties=properties,
                abstraction_level=abstraction_level,
                confidence=0.8,  # Confiança inicial
                examples=[],
                relations=relations or {},
                created_at=time.time(),
                last_updated=time.time()
            )
            
            self.concepts[name] = concept
            
            # Atualizar grafo de conceitos
            self._update_concept_graph(concept)
            
            # Salvar dados
            self._save_data()
            
            return True
            
        except Exception as e:
            logger.error(f"Erro ao adicionar conceito: {e}")
            return False
    
    def _update_concept_graph(self, concept: Concept):
        """Atualiza o grafo de conceitos."""
        for relation_type, related_concepts in concept.relations.items():
            for related_concept in related_concepts:
                self.concept_graph[concept.name].add(related_concept)
                self.concept_graph[related_concept].add(concept.name)
    
    def _initialize_basic_rules(self):
        """Inicializa regras básicas de inferência."""
        basic_rules = [
            {
                "rule_id": "inheritance_rule",
                "premise_pattern": ["X is_a Y", "Y has_property P"],
                "conclusion_pattern": ["X has_property P"],
                "inference_type": InferenceType.DEDUCTIVE,
                "confidence": 0.9
            },
            {
                "rule_id": "similarity_rule",
                "premise_pattern": ["X similar_to Y", "Y has_property P"],
                "conclusion_pattern": ["X likely_has_property P"],
                "inference_type": InferenceType.ANALOGICAL,
                "confidence": 0.7
            },
            {
                "rule_id": "causal_rule",
                "premise_pattern": ["X causes Y", "Y has_property P"],
                "conclusion_pattern": ["X enables_property P"],
                "inference_type": InferenceType.CAUSAL,
                "confidence": 0.8
            }
        ]
        
        for rule_data in basic_rules:
            rule = InferenceRule(
                rule_id=rule_data["rule_id"],
                premise_pattern=rule_data["premise_pattern"],
                conclusion_pattern=rule_data["conclusion_pattern"],
                inference_type=rule_data["inference_type"],
                confidence=rule_data["confidence"],
                usage_count=0,
                success_rate=0.0,
                created_at=time.time(),
                last_updated=time.time()
            )
            self.inference_rules.append(rule)
    
    def get_conceptual_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """Obtém análise conceitual."""
        try:
            if concept_name:
                if concept_name not in self.concepts:
                    return {"error": f"Conceito '{concept_name}' não encontrado"}
                
                concept = self.concepts[concept_name]
                return {
                    "concept": asdict(concept),
                    "related_concepts": list(self.concept_graph.get(concept_name, [])),
                    "inference_rules_applicable": len([
                        rule for rule in self.inference_rules 
                        if concept_name in str(rule.premise_pattern) or concept_name in str(rule.conclusion_pattern)
                    ])
                }
            else:
                # Análise geral
                return {
                    "total_concepts": len(self.concepts),
                    "total_rules": len(self.inference_rules),
                    "total_inferences": len(self.inference_history),
                    "concept_distribution": {
                        level.value: len([c for c in self.concepts.values() if c.abstraction_level == level])
                        for level in AbstractionLevel
                    },
                    "most_common_properties": self._get_most_common_properties(),
                    "concept_graph_stats": {
                        "total_nodes": len(self.concept_graph),
                        "total_edges": sum(len(connections) for connections in self.concept_graph.values()) // 2
                    }
                }
                
        except Exception as e:
            logger.error(f"Erro na análise conceitual: {e}")
            return {"error": str(e)}
    
    def _get_most_common_properties(self) -> List[Tuple[str, int]]:
        """Obtém propriedades mais comuns."""
        property_counts = Counter()
        
        for concept in self.concepts.values():
            for property_name in concept.properties:
                property_counts[property_name] += 1
        
        return property_counts.most_common(10)
    
    def _load_data(self):
        """Carrega dados salvos."""
        try:
            data_file = "data/conceptual_reasoning.json"
            if os.path.exists(data_file):
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Carregar conceitos
                if "concepts" in data:
                    for concept_name, concept_data in data["concepts"].items():
                        concept_data["abstraction_level"] = AbstractionLevel(concept_data["abstraction_level"])
                        self.concepts[concept_name] = Concept(**concept_data)
                
                # Carregar regras de inferência
                if "inference_rules" in data:
                    for rule_data in data["inference_rules"]:
                        rule_data["inference_type"] = InferenceType(rule_data["inference_type"])
                        self.inference_rules.append(InferenceRule(**rule_data))
                
                # Carregar histórico de inferências
                if "inference_history" in data:
                    for result_data in data["inference_history"]:
                        result_data["inference_type"] = InferenceType(result_data["inference_type"])
                        self.inference_history.append(InferenceResult(**result_data))
                
                logger.info(f"Dados de raciocínio conceitual carregados: {len(self.concepts)} conceitos, {len(self.inference_rules)} regras")
                
        except Exception as e:
            logger.warning(f"Erro ao carregar dados de raciocínio conceitual: {e}")
    
    def _save_data(self):
        """Salva dados."""
        try:
            data_file = "data/conceptual_reasoning.json"
            os.makedirs(os.path.dirname(data_file), exist_ok=True)
            
            data = {
                "concepts": {
                    name: {
                        "name": concept.name,
                        "definition": concept.definition,
                        "properties": concept.properties,
                        "abstraction_level": concept.abstraction_level.value,
                        "confidence": concept.confidence,
                        "examples": concept.examples,
                        "relations": concept.relations,
                        "created_at": concept.created_at,
                        "last_updated": concept.last_updated
                    }
                    for name, concept in self.concepts.items()
                },
                "inference_rules": [{
                    "rule_id": rule.rule_id,
                    "premise_pattern": rule.premise_pattern,
                    "conclusion_pattern": rule.conclusion_pattern,
                    "inference_type": rule.inference_type.value,
                    "confidence": rule.confidence,
                    "usage_count": rule.usage_count,
                    "success_rate": rule.success_rate,
                    "created_at": rule.created_at,
                    "last_updated": rule.last_updated
                } for rule in self.inference_rules],
                "inference_history": [{
                    "inference_id": result.inference_id,
                    "premise_concepts": result.premise_concepts,
                    "conclusion_concepts": result.conclusion_concepts,
                    "inference_type": result.inference_type.value,
                    "confidence": result.confidence,
                    "reasoning_steps": result.reasoning_steps,
                    "supporting_evidence": result.supporting_evidence,
                    "timestamp": result.timestamp
                } for result in self.inference_history],
                "last_saved": time.time()
            }
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de raciocínio conceitual salvos: {len(self.concepts)} conceitos, {len(self.inference_rules)} regras")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de raciocínio conceitual: {e}")
