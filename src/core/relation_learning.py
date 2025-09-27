#!/usr/bin/env python3
"""
Sistema de Aprendizado de Relações
Permite ao sistema aprender e inferir relações complexas entre conceitos, objetos e eventos
"""

import logging
import json
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Set, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import random
import time
import hashlib

# Configurar logging
logger = logging.getLogger(__name__)

class RelationType(Enum):
    """Tipos de relações que podem ser aprendidas."""
    TEMPORAL = "temporal"  # Relação temporal (antes, depois, durante)
    SPATIAL = "spatial"    # Relação espacial (dentro, fora, próximo)
    CAUSAL = "causal"      # Relação causal (causa, efeito)
    FUNCTIONAL = "functional"  # Relação funcional (função, propósito)
    STRUCTURAL = "structural"   # Relação estrutural (parte, todo)
    BEHAVIORAL = "behavioral"  # Relação comportamental (ação, reação)
    SIMILARITY = "similarity"  # Relação de similaridade
    OPPOSITION = "opposition"  # Relação de oposição
    DEPENDENCY = "dependency"  # Relação de dependência
    ASSOCIATION = "association"  # Relação de associação

class RelationStrength(Enum):
    """Força das relações."""
    VERY_WEAK = "very_weak"    # 0.0 - 0.2
    WEAK = "weak"              # 0.2 - 0.4
    MODERATE = "moderate"      # 0.4 - 0.6
    STRONG = "strong"          # 0.6 - 0.8
    VERY_STRONG = "very_strong" # 0.8 - 1.0

class LearningMode(Enum):
    """Modos de aprendizado de relações."""
    SUPERVISED = "supervised"      # Aprendizado supervisionado
    UNSUPERVISED = "unsupervised"  # Aprendizado não-supervisionado
    SEMI_SUPERVISED = "semi_supervised"  # Aprendizado semi-supervisionado
    REINFORCEMENT = "reinforcement"  # Aprendizado por reforço
    TRANSFER = "transfer"         # Transferência de conhecimento

class EvidenceType(Enum):
    """Tipos de evidência para relações."""
    DIRECT_OBSERVATION = "direct_observation"  # Observação direta
    STATISTICAL_CORRELATION = "statistical_correlation"  # Correlação estatística
    EXPERT_KNOWLEDGE = "expert_knowledge"      # Conhecimento especializado
    LOGICAL_INFERENCE = "logical_inference"    # Inferência lógica
    PATTERN_RECOGNITION = "pattern_recognition"  # Reconhecimento de padrões
    ANALOGICAL_REASONING = "analogical_reasoning"  # Raciocínio analógico

@dataclass
class RelationEvidence:
    """Evidência para uma relação."""
    evidence_id: str
    evidence_type: EvidenceType
    source: str  # Fonte da evidência
    confidence: float  # Confiança na evidência (0.0 a 1.0)
    context: Dict[str, Any]  # Contexto da evidência
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LearnedRelation:
    """Relação aprendida pelo sistema."""
    relation_id: str
    source_concept: str
    target_concept: str
    relation_type: RelationType
    strength: RelationStrength
    confidence: float  # Confiança geral na relação
    evidence: List[RelationEvidence]  # Evidências que suportam a relação
    learning_mode: LearningMode
    context: Dict[str, Any]  # Contexto da relação
    frequency: int = 0  # Frequência de observação
    last_observed: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RelationPattern:
    """Padrão de relações identificado."""
    pattern_id: str
    pattern_type: str
    relations: List[str]  # IDs das relações que formam o padrão
    frequency: int
    confidence: float
    context: Dict[str, Any]
    created_at: float = field(default_factory=time.time)

class RelationLearningSystem:
    """Sistema de aprendizado de relações."""
    
    def __init__(self):
        self.learned_relations: Dict[str, LearnedRelation] = {}
        self.relation_patterns: Dict[str, RelationPattern] = {}
        self.evidence_base: Dict[str, List[RelationEvidence]] = defaultdict(list)
        self.relation_graph: Dict[str, Set[str]] = defaultdict(set)
        self.learning_history: List[Dict[str, Any]] = []
        
        # Carregar dados existentes
        self._load_data()
        
        # Inicializar relações básicas
        self._initialize_basic_relations()
        
        logger.info("Sistema de aprendizado de relações inicializado")
    
    def learn_relation(self, source_concept: str, target_concept: str,
                      relation_type: RelationType, evidence: List[RelationEvidence],
                      learning_mode: LearningMode = LearningMode.UNSUPERVISED,
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Aprende uma nova relação entre conceitos.
        
        Args:
            source_concept: Conceito origem
            target_concept: Conceito destino
            relation_type: Tipo da relação
            evidence: Evidências que suportam a relação
            learning_mode: Modo de aprendizado
            context: Contexto da relação
            
        Returns:
            Resultado do aprendizado
        """
        try:
            # Gerar ID único para a relação
            relation_id = self._generate_relation_id(source_concept, target_concept, relation_type)
            
            # Verificar se a relação já existe
            if relation_id in self.learned_relations:
                return self._update_existing_relation(relation_id, evidence, context)
            
            # Calcular força da relação baseada nas evidências
            strength = self._calculate_relation_strength(evidence)
            
            # Calcular confiança geral
            confidence = self._calculate_relation_confidence(evidence)
            
            # Criar relação aprendida
            relation = LearnedRelation(
                relation_id=relation_id,
                source_concept=source_concept,
                target_concept=target_concept,
                relation_type=relation_type,
                strength=strength,
                confidence=confidence,
                evidence=evidence,
                learning_mode=learning_mode,
                context=context or {},
                frequency=1,
                last_observed=time.time()
            )
            
            # Armazenar relação
            self.learned_relations[relation_id] = relation
            
            # Atualizar grafo de relações
            self._update_relation_graph(relation)
            
            # Armazenar evidências
            self._store_evidence(relation_id, evidence)
            
            # Detectar padrões
            self._detect_relation_patterns(relation)
            
            # Registrar aprendizado
            self._record_learning_event(relation, "learn")
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "relation_id": relation_id,
                "strength": strength.value,
                "confidence": confidence,
                "message": f"Relação {relation_type.value} aprendida entre {source_concept} e {target_concept}"
            }
            
        except Exception as e:
            logger.error(f"Erro ao aprender relação: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def infer_relation(self, source_concept: str, target_concept: str,
                      relation_type: RelationType = None,
                      context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Infere uma relação entre conceitos baseada no conhecimento existente.
        
        Args:
            source_concept: Conceito origem
            target_concept: Conceito destino
            relation_type: Tipo de relação específico (opcional)
            context: Contexto da inferência
            
        Returns:
            Resultado da inferência
        """
        try:
            # Buscar relações existentes
            existing_relations = self._find_existing_relations(
                source_concept, target_concept, relation_type
            )
            
            if existing_relations:
                # Retornar relação mais forte
                best_relation = max(existing_relations, key=lambda r: r.confidence)
                return {
                    "success": True,
                    "relation_found": True,
                    "relation_id": best_relation.relation_id,
                    "relation_type": best_relation.relation_type.value,
                    "strength": best_relation.strength.value,
                    "confidence": best_relation.confidence,
                    "evidence_count": len(best_relation.evidence)
                }
            
            # Tentar inferir nova relação
            inferred_relation = self._infer_new_relation(
                source_concept, target_concept, relation_type, context
            )
            
            if inferred_relation:
                return {
                    "success": True,
                    "relation_found": False,
                    "inferred_relation": inferred_relation,
                    "confidence": inferred_relation.get("confidence", 0.0)
                }
            
            return {
                "success": True,
                "relation_found": False,
                "message": "Nenhuma relação encontrada ou inferida"
            }
            
        except Exception as e:
            logger.error(f"Erro na inferência de relação: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def validate_relation(self, relation_id: str, new_evidence: List[RelationEvidence]) -> Dict[str, Any]:
        """
        Valida uma relação existente com novas evidências.
        
        Args:
            relation_id: ID da relação
            new_evidence: Novas evidências
            
        Returns:
            Resultado da validação
        """
        try:
            if relation_id not in self.learned_relations:
                return {
                    "success": False,
                    "error": "Relação não encontrada"
                }
            
            relation = self.learned_relations[relation_id]
            
            # Adicionar novas evidências
            relation.evidence.extend(new_evidence)
            
            # Recalcular força e confiança
            old_strength = relation.strength
            old_confidence = relation.confidence
            
            relation.strength = self._calculate_relation_strength(relation.evidence)
            relation.confidence = self._calculate_relation_confidence(relation.evidence)
            relation.frequency += 1
            relation.last_observed = time.time()
            relation.last_updated = time.time()
            
            # Registrar validação
            self._record_learning_event(relation, "validate", {
                "old_strength": old_strength.value,
                "new_strength": relation.strength.value,
                "old_confidence": old_confidence,
                "new_confidence": relation.confidence,
                "new_evidence_count": len(new_evidence)
            })
            
            # Salvar dados
            self._save_data()
            
            return {
                "success": True,
                "relation_id": relation_id,
                "strength_change": {
                    "old": old_strength.value,
                    "new": relation.strength.value
                },
                "confidence_change": {
                    "old": old_confidence,
                    "new": relation.confidence
                },
                "total_evidence": len(relation.evidence)
            }
            
        except Exception as e:
            logger.error(f"Erro na validação de relação: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def find_relation_patterns(self, concept: str = None, 
                             pattern_type: str = None) -> Dict[str, Any]:
        """
        Encontra padrões de relações.
        
        Args:
            concept: Conceito específico (opcional)
            pattern_type: Tipo de padrão (opcional)
            
        Returns:
            Padrões encontrados
        """
        try:
            patterns = []
            
            for pattern in self.relation_patterns.values():
                if concept and concept not in self._get_pattern_concepts(pattern):
                    continue
                if pattern_type and pattern.pattern_type != pattern_type:
                    continue
                
                patterns.append({
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "relations": pattern.relations,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "context": pattern.context
                })
            
            return {
                "success": True,
                "patterns": patterns,
                "count": len(patterns)
            }
            
        except Exception as e:
            logger.error(f"Erro ao encontrar padrões: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_relation_analysis(self, concept: str = None) -> Dict[str, Any]:
        """
        Obtém análise das relações aprendidas.
        
        Args:
            concept: Conceito específico (opcional)
            
        Returns:
            Análise das relações
        """
        try:
            if concept:
                # Análise de um conceito específico
                relations = [r for r in self.learned_relations.values() 
                           if r.source_concept == concept or r.target_concept == concept]
            else:
                # Análise geral
                relations = list(self.learned_relations.values())
            
            # Estatísticas gerais
            total_relations = len(relations)
            relation_types = defaultdict(int)
            strength_distribution = defaultdict(int)
            learning_modes = defaultdict(int)
            
            for relation in relations:
                relation_types[relation.relation_type.value] += 1
                strength_distribution[relation.strength.value] += 1
                learning_modes[relation.learning_mode.value] += 1
            
            # Conceitos mais relacionados
            concept_frequency = defaultdict(int)
            for relation in relations:
                concept_frequency[relation.source_concept] += 1
                concept_frequency[relation.target_concept] += 1
            
            most_related = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Padrões mais frequentes
            pattern_frequency = defaultdict(int)
            for pattern in self.relation_patterns.values():
                pattern_frequency[pattern.pattern_type] += pattern.frequency
            
            most_frequent_patterns = sorted(pattern_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "success": True,
                "analysis": {
                    "total_relations": total_relations,
                    "relation_types": dict(relation_types),
                    "strength_distribution": dict(strength_distribution),
                    "learning_modes": dict(learning_modes),
                    "most_related_concepts": most_related,
                    "most_frequent_patterns": most_frequent_patterns,
                    "total_patterns": len(self.relation_patterns),
                    "total_evidence": sum(len(r.evidence) for r in relations)
                }
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de relações: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _generate_relation_id(self, source: str, target: str, relation_type: RelationType) -> str:
        """Gera ID único para uma relação."""
        content = f"{source}_{target}_{relation_type.value}"
        return f"rel_{hashlib.md5(content.encode()).hexdigest()[:12]}"
    
    def _calculate_relation_strength(self, evidence: List[RelationEvidence]) -> RelationStrength:
        """Calcula força da relação baseada nas evidências."""
        if not evidence:
            return RelationStrength.WEAK
        
        # Calcular força média ponderada pela confiança
        total_weight = sum(e.confidence for e in evidence)
        if total_weight == 0:
            return RelationStrength.WEAK
        
        weighted_strength = sum(e.confidence * self._get_evidence_strength(e) for e in evidence) / total_weight
        
        # Converter para enum
        if weighted_strength >= 0.8:
            return RelationStrength.VERY_STRONG
        elif weighted_strength >= 0.6:
            return RelationStrength.STRONG
        elif weighted_strength >= 0.4:
            return RelationStrength.MODERATE
        elif weighted_strength >= 0.2:
            return RelationStrength.WEAK
        else:
            return RelationStrength.VERY_WEAK
    
    def _calculate_relation_confidence(self, evidence: List[RelationEvidence]) -> float:
        """Calcula confiança geral da relação."""
        if not evidence:
            return 0.0
        
        # Confiança baseada na quantidade e qualidade das evidências
        evidence_count_factor = min(len(evidence) / 5.0, 1.0)  # Máximo em 5 evidências
        evidence_quality_factor = sum(e.confidence for e in evidence) / len(evidence)
        
        # Diversidade de tipos de evidência
        evidence_types = set(e.evidence_type for e in evidence)
        diversity_factor = min(len(evidence_types) / 3.0, 1.0)  # Máximo em 3 tipos
        
        confidence = (evidence_count_factor * 0.3 + 
                     evidence_quality_factor * 0.5 + 
                     diversity_factor * 0.2)
        
        return min(confidence, 1.0)
    
    def _get_evidence_strength(self, evidence: RelationEvidence) -> float:
        """Obtém força de uma evidência baseada no tipo."""
        strength_map = {
            EvidenceType.DIRECT_OBSERVATION: 0.9,
            EvidenceType.STATISTICAL_CORRELATION: 0.7,
            EvidenceType.EXPERT_KNOWLEDGE: 0.8,
            EvidenceType.LOGICAL_INFERENCE: 0.6,
            EvidenceType.PATTERN_RECOGNITION: 0.5,
            EvidenceType.ANALOGICAL_REASONING: 0.4
        }
        return strength_map.get(evidence.evidence_type, 0.5)
    
    def _update_existing_relation(self, relation_id: str, evidence: List[RelationEvidence], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Atualiza relação existente com novas evidências."""
        relation = self.learned_relations[relation_id]
        
        # Adicionar novas evidências
        relation.evidence.extend(evidence)
        
        # Recalcular métricas
        old_strength = relation.strength
        old_confidence = relation.confidence
        
        relation.strength = self._calculate_relation_strength(relation.evidence)
        relation.confidence = self._calculate_relation_confidence(relation.evidence)
        relation.frequency += 1
        relation.last_observed = time.time()
        relation.last_updated = time.time()
        
        # Atualizar contexto
        if context:
            relation.context.update(context)
        
        return {
            "success": True,
            "relation_id": relation_id,
            "updated": True,
            "strength_change": {
                "old": old_strength.value,
                "new": relation.strength.value
            },
            "confidence_change": {
                "old": old_confidence,
                "new": relation.confidence
            }
        }
    
    def _find_existing_relations(self, source: str, target: str, 
                               relation_type: RelationType = None) -> List[LearnedRelation]:
        """Encontra relações existentes entre conceitos."""
        relations = []
        
        for relation in self.learned_relations.values():
            if ((relation.source_concept == source and relation.target_concept == target) or
                (relation.source_concept == target and relation.target_concept == source)):
                
                if relation_type is None or relation.relation_type == relation_type:
                    relations.append(relation)
        
        return relations
    
    def _infer_new_relation(self, source: str, target: str, 
                           relation_type: RelationType = None,
                           context: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Tenta inferir uma nova relação."""
        # Implementar lógica de inferência baseada em padrões existentes
        # Por enquanto, retornar None (não implementado)
        return None
    
    def _update_relation_graph(self, relation: LearnedRelation):
        """Atualiza grafo de relações."""
        self.relation_graph[relation.source_concept].add(relation.target_concept)
        self.relation_graph[relation.target_concept].add(relation.source_concept)
    
    def _store_evidence(self, relation_id: str, evidence: List[RelationEvidence]):
        """Armazena evidências."""
        self.evidence_base[relation_id].extend(evidence)
    
    def _detect_relation_patterns(self, relation: LearnedRelation):
        """Detecta padrões de relações."""
        # Implementar detecção de padrões
        # Por enquanto, não implementado
        pass
    
    def _record_learning_event(self, relation: LearnedRelation, event_type: str, 
                              metadata: Dict[str, Any] = None):
        """Registra evento de aprendizado."""
        event = {
            "timestamp": time.time(),
            "event_type": event_type,
            "relation_id": relation.relation_id,
            "source_concept": relation.source_concept,
            "target_concept": relation.target_concept,
            "relation_type": relation.relation_type.value,
            "metadata": metadata or {}
        }
        self.learning_history.append(event)
    
    def _get_pattern_concepts(self, pattern: RelationPattern) -> Set[str]:
        """Obtém conceitos envolvidos em um padrão."""
        concepts = set()
        for relation_id in pattern.relations:
            if relation_id in self.learned_relations:
                relation = self.learned_relations[relation_id]
                concepts.add(relation.source_concept)
                concepts.add(relation.target_concept)
        return concepts
    
    def _initialize_basic_relations(self):
        """Inicializa relações básicas."""
        # Relações básicas entre conceitos de pássaros
        basic_relations = [
            ("pássaro", "ave", RelationType.STRUCTURAL, [
                RelationEvidence(
                    evidence_id="ev_001",
                    evidence_type=EvidenceType.EXPERT_KNOWLEDGE,
                    source="taxonomia",
                    confidence=0.9,
                    context={"domain": "biology"},
                    timestamp=time.time()
                )
            ]),
            ("ave", "voo", RelationType.FUNCTIONAL, [
                RelationEvidence(
                    evidence_id="ev_002",
                    evidence_type=EvidenceType.DIRECT_OBSERVATION,
                    source="observação",
                    confidence=0.8,
                    context={"domain": "behavior"},
                    timestamp=time.time()
                )
            ]),
            ("pássaro", "penas", RelationType.STRUCTURAL, [
                RelationEvidence(
                    evidence_id="ev_003",
                    evidence_type=EvidenceType.EXPERT_KNOWLEDGE,
                    source="anatomia",
                    confidence=0.95,
                    context={"domain": "morphology"},
                    timestamp=time.time()
                )
            ])
        ]
        
        for source, target, rel_type, evidence in basic_relations:
            self.learn_relation(source, target, rel_type, evidence)
    
    def _save_data(self):
        """Salva dados do sistema."""
        try:
            data = {
                "learned_relations": {},
                "relation_patterns": {},
                "learning_history": self.learning_history[-100:]  # Últimos 100 eventos
            }
            
            # Converter relações para formato serializável
            for rel_id, relation in self.learned_relations.items():
                data["learned_relations"][rel_id] = {
                    "relation_id": relation.relation_id,
                    "source_concept": relation.source_concept,
                    "target_concept": relation.target_concept,
                    "relation_type": relation.relation_type.value,
                    "strength": relation.strength.value,
                    "confidence": relation.confidence,
                    "learning_mode": relation.learning_mode.value,
                    "context": relation.context,
                    "frequency": relation.frequency,
                    "last_observed": relation.last_observed,
                    "created_at": relation.created_at,
                    "last_updated": relation.last_updated,
                    "metadata": relation.metadata,
                    "evidence": [
                        {
                            "evidence_id": e.evidence_id,
                            "evidence_type": e.evidence_type.value,
                            "source": e.source,
                            "confidence": e.confidence,
                            "context": e.context,
                            "timestamp": e.timestamp,
                            "metadata": e.metadata
                        } for e in relation.evidence
                    ]
                }
            
            # Converter padrões para formato serializável
            for pattern_id, pattern in self.relation_patterns.items():
                data["relation_patterns"][pattern_id] = {
                    "pattern_id": pattern.pattern_id,
                    "pattern_type": pattern.pattern_type,
                    "relations": pattern.relations,
                    "frequency": pattern.frequency,
                    "confidence": pattern.confidence,
                    "context": pattern.context,
                    "created_at": pattern.created_at
                }
            
            # Salvar arquivo
            os.makedirs("data", exist_ok=True)
            with open("data/relation_learning_data.json", "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Dados de aprendizado de relações salvos: {len(self.learned_relations)} relações, {len(self.relation_patterns)} padrões")
            
        except Exception as e:
            logger.error(f"Erro ao salvar dados de aprendizado de relações: {e}")
    
    def _load_data(self):
        """Carrega dados do sistema."""
        try:
            if not os.path.exists("data/relation_learning_data.json"):
                return
            
            with open("data/relation_learning_data.json", "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Carregar relações
            for rel_id, rel_data in data.get("learned_relations", {}).items():
                evidence = [
                    RelationEvidence(
                        evidence_id=e["evidence_id"],
                        evidence_type=EvidenceType(e["evidence_type"]),
                        source=e["source"],
                        confidence=e["confidence"],
                        context=e["context"],
                        timestamp=e["timestamp"],
                        metadata=e.get("metadata", {})
                    ) for e in rel_data.get("evidence", [])
                ]
                
                relation = LearnedRelation(
                    relation_id=rel_data["relation_id"],
                    source_concept=rel_data["source_concept"],
                    target_concept=rel_data["target_concept"],
                    relation_type=RelationType(rel_data["relation_type"]),
                    strength=RelationStrength(rel_data["strength"]),
                    confidence=rel_data["confidence"],
                    evidence=evidence,
                    learning_mode=LearningMode(rel_data["learning_mode"]),
                    context=rel_data["context"],
                    frequency=rel_data["frequency"],
                    last_observed=rel_data["last_observed"],
                    created_at=rel_data["created_at"],
                    last_updated=rel_data["last_updated"],
                    metadata=rel_data.get("metadata", {})
                )
                
                self.learned_relations[rel_id] = relation
                self._update_relation_graph(relation)
            
            # Carregar padrões
            for pattern_id, pattern_data in data.get("relation_patterns", {}).items():
                pattern = RelationPattern(
                    pattern_id=pattern_data["pattern_id"],
                    pattern_type=pattern_data["pattern_type"],
                    relations=pattern_data["relations"],
                    frequency=pattern_data["frequency"],
                    confidence=pattern_data["confidence"],
                    context=pattern_data["context"],
                    created_at=pattern_data["created_at"]
                )
                self.relation_patterns[pattern_id] = pattern
            
            # Carregar histórico
            self.learning_history = data.get("learning_history", [])
            
            logger.info(f"Dados de aprendizado de relações carregados: {len(self.learned_relations)} relações, {len(self.relation_patterns)} padrões")
            
        except Exception as e:
            logger.error(f"Erro ao carregar dados de aprendizado de relações: {e}")
