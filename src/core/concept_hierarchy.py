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
import networkx as nx

# Configurar logging
logger = logging.getLogger(__name__)

class ConceptRelationshipType(Enum):
    """Tipos de relacionamentos entre conceitos."""
    IS_A = "is_a"  # Relacionamento de herança (é um tipo de)
    PART_OF = "part_of"  # Relacionamento de composição (é parte de)
    HAS_A = "has_a"  # Relacionamento de posse (tem um)
    SIMILAR_TO = "similar_to"  # Relacionamento de similaridade
    OPPOSITE_OF = "opposite_of"  # Relacionamento de oposição
    CAUSES = "causes"  # Relacionamento causal
    REQUIRES = "requires"  # Relacionamento de dependência
    INHIBITS = "inhibits"  # Relacionamento de inibição

class ConceptAbstractionLevel(Enum):
    """Níveis de abstração dos conceitos."""
    CONCRETE = "concrete"  # Conceitos concretos (objetos específicos)
    SPECIFIC = "specific"  # Conceitos específicos (espécies)
    GENERAL = "general"  # Conceitos gerais (gêneros)
    ABSTRACT = "abstract"  # Conceitos abstratos (famílias)
    UNIVERSAL = "universal"  # Conceitos universais (classes)
    METAPHYSICAL = "metaphysical"  # Conceitos metafísicos (reinos)

class ConceptComplexity(Enum):
    """Níveis de complexidade dos conceitos."""
    SIMPLE = "simple"  # Conceitos simples (uma característica)
    COMPOUND = "compound"  # Conceitos compostos (múltiplas características)
    COMPLEX = "complex"  # Conceitos complexos (múltiplas dimensões)
    META = "meta"  # Conceitos meta (sobre conceitos)

@dataclass
class ConceptRelationship:
    """Representa um relacionamento entre conceitos."""
    source_concept: str
    target_concept: str
    relationship_type: ConceptRelationshipType
    strength: float  # Força do relacionamento (0.0 a 1.0)
    confidence: float  # Confiança no relacionamento (0.0 a 1.0)
    evidence: List[str] = field(default_factory=list)  # Evidências do relacionamento
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)

@dataclass
class HierarchicalConcept:
    """Representa um conceito hierárquico."""
    name: str
    definition: str
    abstraction_level: ConceptAbstractionLevel
    complexity: ConceptComplexity
    parent_concepts: List[str] = field(default_factory=list)
    child_concepts: List[str] = field(default_factory=list)
    characteristics: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    counter_examples: List[str] = field(default_factory=list)
    relationships: List[ConceptRelationship] = field(default_factory=list)
    confidence_score: float = 0.0
    usage_frequency: int = 0
    last_used: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

class ConceptHierarchyManager:
    """Gerenciador de hierarquias de conceitos."""
    
    def __init__(self):
        self.concepts: Dict[str, HierarchicalConcept] = {}
        self.relationships: List[ConceptRelationship] = []
        self.concept_graph = nx.DiGraph()  # Grafo direcionado para hierarquias
        self.similarity_graph = nx.Graph()  # Grafo não-direcionado para similaridades
        
        # Inicializar conceitos base
        self._initialize_base_concepts()
        
        logger.info("Sistema de hierarquias de conceitos inicializado")
    
    def _initialize_base_concepts(self):
        """Inicializa conceitos base da hierarquia."""
        
        # Conceitos universais (nível mais alto)
        self._add_concept("animal", 
                         "Ser vivo multicelular que se move e se alimenta",
                         ConceptAbstractionLevel.UNIVERSAL,
                         ConceptComplexity.COMPOUND)
        
        self._add_concept("bird", 
                         "Animal vertebrado com penas, bico e capacidade de voar",
                         ConceptAbstractionLevel.ABSTRACT,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["animal"])
        
        self._add_concept("mammal", 
                         "Animal vertebrado com pelos e glândulas mamárias",
                         ConceptAbstractionLevel.ABSTRACT,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["animal"])
        
        # Conceitos específicos de pássaros
        self._add_concept("passerine", 
                         "Pássaro da ordem Passeriformes, com três dedos para frente",
                         ConceptAbstractionLevel.GENERAL,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["bird"])
        
        self._add_concept("raptor", 
                         "Pássaro de rapina com garras afiadas e bico curvo",
                         ConceptAbstractionLevel.GENERAL,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["bird"])
        
        self._add_concept("waterfowl", 
                         "Pássaro aquático com patas palmadas",
                         ConceptAbstractionLevel.GENERAL,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["bird"])
        
        # Conceitos específicos de espécies
        self._add_concept("blue_jay", 
                         "Pássaro azul com crista e padrões brancos",
                         ConceptAbstractionLevel.SPECIFIC,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["passerine"])
        
        self._add_concept("eagle", 
                         "Pássaro de rapina grande com asas largas",
                         ConceptAbstractionLevel.SPECIFIC,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["raptor"])
        
        self._add_concept("duck", 
                         "Pássaro aquático com bico largo e patas palmadas",
                         ConceptAbstractionLevel.SPECIFIC,
                         ConceptComplexity.COMPOUND,
                         parent_concepts=["waterfowl"])
        
        # Conceitos de características
        self._add_concept("feather", 
                         "Estrutura de queratina que cobre o corpo de pássaros",
                         ConceptAbstractionLevel.CONCRETE,
                         ConceptComplexity.SIMPLE)
        
        self._add_concept("beak", 
                         "Estrutura córnea na boca de pássaros",
                         ConceptAbstractionLevel.CONCRETE,
                         ConceptComplexity.SIMPLE)
        
        self._add_concept("wing", 
                         "Estrutura para voo em pássaros",
                         ConceptAbstractionLevel.CONCRETE,
                         ConceptComplexity.SIMPLE)
        
        # Conceitos comportamentais
        self._add_concept("flying", 
                         "Ato de se mover pelo ar usando asas",
                         ConceptAbstractionLevel.ABSTRACT,
                         ConceptComplexity.COMPOUND)
        
        self._add_concept("singing", 
                         "Produção de sons musicais por pássaros",
                         ConceptAbstractionLevel.ABSTRACT,
                         ConceptComplexity.COMPOUND)
        
        self._add_concept("nesting", 
                         "Construção de ninho para reprodução",
                         ConceptAbstractionLevel.ABSTRACT,
                         ConceptComplexity.COMPOUND)
        
        # Estabelecer relacionamentos
        self._establish_relationships()
    
    def _add_concept(self, name: str, definition: str, abstraction_level: ConceptAbstractionLevel, 
                    complexity: ConceptComplexity, parent_concepts: List[str] = None):
        """Adiciona um conceito à hierarquia."""
        concept = HierarchicalConcept(
            name=name,
            definition=definition,
            abstraction_level=abstraction_level,
            complexity=complexity,
            parent_concepts=parent_concepts or [],
            characteristics=[],
            examples=[],
            counter_examples=[],
            confidence_score=0.5
        )
        
        self.concepts[name] = concept
        self.concept_graph.add_node(name, concept=concept)
        
        # Adicionar arestas para pais
        if parent_concepts:
            for parent in parent_concepts:
                if parent in self.concepts:
                    self.concept_graph.add_edge(parent, name)
                    self.concepts[parent].child_concepts.append(name)
        
        return name  # Retornar o nome como ID
    
    def _establish_relationships(self):
        """Estabelece relacionamentos entre conceitos."""
        
        # Relacionamentos de herança (IS_A)
        inheritance_relationships = [
            ("bird", "animal", ConceptRelationshipType.IS_A, 1.0),
            ("mammal", "animal", ConceptRelationshipType.IS_A, 1.0),
            ("passerine", "bird", ConceptRelationshipType.IS_A, 1.0),
            ("raptor", "bird", ConceptRelationshipType.IS_A, 1.0),
            ("waterfowl", "bird", ConceptRelationshipType.IS_A, 1.0),
            ("blue_jay", "passerine", ConceptRelationshipType.IS_A, 1.0),
            ("eagle", "raptor", ConceptRelationshipType.IS_A, 1.0),
            ("duck", "waterfowl", ConceptRelationshipType.IS_A, 1.0)
        ]
        
        for source, target, rel_type, strength in inheritance_relationships:
            self._add_relationship(source, target, rel_type, strength, 1.0)
        
        # Relacionamentos de composição (HAS_A)
        composition_relationships = [
            ("bird", "feather", ConceptRelationshipType.HAS_A, 1.0),
            ("bird", "beak", ConceptRelationshipType.HAS_A, 1.0),
            ("bird", "wing", ConceptRelationshipType.HAS_A, 1.0),
            ("passerine", "feather", ConceptRelationshipType.HAS_A, 1.0),
            ("raptor", "beak", ConceptRelationshipType.HAS_A, 1.0),
            ("waterfowl", "wing", ConceptRelationshipType.HAS_A, 1.0)
        ]
        
        for source, target, rel_type, strength in composition_relationships:
            self._add_relationship(source, target, rel_type, strength, 1.0)
        
        # Relacionamentos comportamentais (CAUSES)
        behavioral_relationships = [
            ("wing", "flying", ConceptRelationshipType.CAUSES, 0.9),
            ("bird", "singing", ConceptRelationshipType.CAUSES, 0.7),
            ("bird", "nesting", ConceptRelationshipType.CAUSES, 0.8)
        ]
        
        for source, target, rel_type, strength in behavioral_relationships:
            self._add_relationship(source, target, rel_type, strength, 0.8)
        
        # Relacionamentos de similaridade
        similarity_relationships = [
            ("passerine", "raptor", ConceptRelationshipType.SIMILAR_TO, 0.6),
            ("waterfowl", "raptor", ConceptRelationshipType.SIMILAR_TO, 0.5),
            ("blue_jay", "eagle", ConceptRelationshipType.SIMILAR_TO, 0.4)
        ]
        
        for source, target, rel_type, strength in similarity_relationships:
            self._add_relationship(source, target, rel_type, strength, 0.6)
            self.similarity_graph.add_edge(source, target, weight=strength)
    
    def _add_relationship(self, source: str, target: str, relationship_type: ConceptRelationshipType, 
                         strength: float, confidence: float, evidence: List[str] = None):
        """Adiciona um relacionamento entre conceitos."""
        relationship = ConceptRelationship(
            source_concept=source,
            target_concept=target,
            relationship_type=relationship_type,
            strength=strength,
            confidence=confidence,
            evidence=evidence or []
        )
        
        self.relationships.append(relationship)
        
        # Atualizar conceitos
        if source in self.concepts:
            self.concepts[source].relationships.append(relationship)
    
    def add_concept(self, name: str, description: str, abstraction_level: ConceptAbstractionLevel,
                   complexity: ConceptComplexity, parent_concepts: List[str] = None,
                   properties: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Adiciona um conceito à hierarquia.
        
        Args:
            name: Nome do conceito
            description: Descrição do conceito
            abstraction_level: Nível de abstração
            complexity: Nível de complexidade
            parent_concepts: Conceitos pais (opcional)
            properties: Propriedades do conceito (opcional)
            
        Returns:
            Resultado da adição do conceito
        """
        try:
            concept_id = self._add_concept(name, description, abstraction_level, complexity, parent_concepts)
            return {
                "success": True,
                "concept_id": concept_id,
                "message": f"Conceito '{name}' adicionado com sucesso"
            }
        except Exception as e:
            logger.error(f"Erro ao adicionar conceito '{name}': {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def add_relationship(self, source_concept: str, target_concept: str,
                        relationship_type: ConceptRelationshipType, strength: float = 0.5,
                        confidence: float = 0.5) -> Dict[str, Any]:
        """
        Adiciona um relacionamento entre conceitos.
        
        Args:
            source_concept: Conceito origem
            target_concept: Conceito destino
            relationship_type: Tipo de relacionamento
            strength: Força do relacionamento
            confidence: Confiança no relacionamento
            
        Returns:
            Resultado da adição do relacionamento
        """
        try:
            relationship_id = self._add_relationship(source_concept, target_concept, relationship_type, strength, confidence)
            return {
                "success": True,
                "relationship_id": relationship_id,
                "message": f"Relacionamento entre '{source_concept}' e '{target_concept}' adicionado"
            }
        except Exception as e:
            logger.error(f"Erro ao adicionar relacionamento: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def find_relationships(self, concept_name: str, relationship_type: str = None) -> Dict[str, Any]:
        """
        Encontra relacionamentos de um conceito.
        
        Args:
            concept_name: Nome do conceito
            relationship_type: Tipo de relacionamento (opcional)
            
        Returns:
            Relacionamentos encontrados
        """
        try:
            relationships = []
            for rel in self.relationships:
                if rel.source_concept == concept_name or rel.target_concept == concept_name:
                    if relationship_type is None or rel.relationship_type.value == relationship_type:
                        relationships.append({
                            "source": rel.source_concept,
                            "target": rel.target_concept,
                            "type": rel.relationship_type.value,
                            "strength": rel.strength,
                            "confidence": rel.confidence
                        })
            
            return {
                "success": True,
                "relationships": relationships,
                "count": len(relationships)
            }
        except Exception as e:
            logger.error(f"Erro ao encontrar relacionamentos: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_hierarchy(self, root_concept: str = None, max_depth: int = 3) -> Dict[str, Any]:
        """
        Obtém a hierarquia de conceitos.
        
        Args:
            root_concept: Conceito raiz (opcional)
            max_depth: Profundidade máxima
            
        Returns:
            Hierarquia de conceitos
        """
        try:
            concepts = []
            if root_concept:
                # Obter hierarquia a partir de um conceito específico
                ancestors = self._find_ancestors(root_concept)
                descendants = self._find_descendants(root_concept)
                concepts = ancestors + descendants
            else:
                # Obter todos os conceitos
                concepts = list(self.concepts.values())
            
            # Converter para formato consistente
            concept_list = []
            for c in concepts:
                if isinstance(c, dict):
                    concept_list.append(c)
                else:
                    concept_list.append({
                        "name": c.name, 
                        "description": c.definition, 
                        "abstraction_level": c.abstraction_level.value,
                        "complexity": c.complexity.value
                    })
            
            return {
                "success": True,
                "concepts": concept_list,
                "count": len(concept_list)
            }
        except Exception as e:
            logger.error(f"Erro ao obter hierarquia: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_similarity(self, concept1: str, concept2: str) -> Dict[str, Any]:
        """
        Analisa similaridade entre dois conceitos.
        
        Args:
            concept1: Primeiro conceito
            concept2: Segundo conceito
            
        Returns:
            Análise de similaridade
        """
        try:
            if concept1 not in self.concepts or concept2 not in self.concepts:
                return {
                    "success": False,
                    "error": "Conceito não encontrado"
                }
            
            # Calcular similaridade baseada em características
            c1 = self.concepts[concept1]
            c2 = self.concepts[concept2]
            
            # Similaridade por características
            char_similarity = 0.0
            if c1.characteristics and c2.characteristics:
                char1 = set(c1.characteristics)
                char2 = set(c2.characteristics)
                if char1 or char2:
                    char_similarity = len(char1.intersection(char2)) / len(char1.union(char2))
            
            # Similaridade por nível de abstração
            abstraction_similarity = 1.0 if c1.abstraction_level == c2.abstraction_level else 0.5
            
            # Similaridade por complexidade
            complexity_similarity = 1.0 if c1.complexity == c2.complexity else 0.5
            
            # Similaridade geral
            overall_similarity = (char_similarity + abstraction_similarity + complexity_similarity) / 3
            
            return {
                "success": True,
                "similarity_score": overall_similarity,
                "characteristic_similarity": char_similarity,
                "abstraction_similarity": abstraction_similarity,
                "complexity_similarity": complexity_similarity
            }
        except Exception as e:
            logger.error(f"Erro na análise de similaridade: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_analysis(self, concept_name: str = None) -> Dict[str, Any]:
        """
        Obtém análise da hierarquia de conceitos.
        
        Args:
            concept_name: Nome do conceito específico (opcional)
            
        Returns:
            Análise da hierarquia
        """
        try:
            if concept_name:
                return self.analyze_concept_hierarchy(concept_name)
            else:
                return self.get_hierarchy_statistics()
        except Exception as e:
            logger.error(f"Erro na análise: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def analyze_concept_hierarchy(self, concept_name: str) -> Dict[str, Any]:
        """
        Analisa a hierarquia de um conceito específico.
        
        Args:
            concept_name: Nome do conceito a ser analisado
            
        Returns:
            Análise da hierarquia do conceito
        """
        if concept_name not in self.concepts:
            return {"error": f"Conceito '{concept_name}' não encontrado"}
        
        concept = self.concepts[concept_name]
        
        # Encontrar ancestrais (pais, avós, etc.)
        ancestors = self._find_ancestors(concept_name)
        
        # Encontrar descendentes (filhos, netos, etc.)
        descendants = self._find_descendants(concept_name)
        
        # Encontrar conceitos relacionados
        related_concepts = self._find_related_concepts(concept_name)
        
        # Calcular profundidade na hierarquia
        hierarchy_depth = self._calculate_hierarchy_depth(concept_name)
        
        # Calcular centralidade conceitual
        centrality = self._calculate_concept_centrality(concept_name)
        
        # Análise de características herdadas
        inherited_characteristics = self._get_inherited_characteristics(concept_name)
        
        # Análise de exemplos e contra-exemplos
        examples_analysis = self._analyze_examples(concept_name)
        
        return {
            "concept_name": concept_name,
            "definition": concept.definition,
            "abstraction_level": concept.abstraction_level.value,
            "complexity": concept.complexity.value,
            "confidence_score": concept.confidence_score,
            "usage_frequency": concept.usage_frequency,
            "hierarchy_depth": hierarchy_depth,
            "centrality": centrality,
            "ancestors": ancestors,
            "descendants": descendants,
            "related_concepts": related_concepts,
            "inherited_characteristics": inherited_characteristics,
            "examples_analysis": examples_analysis,
            "relationships_count": len(concept.relationships),
            "total_relationships": len(self.relationships)
        }
    
    def _find_ancestors(self, concept_name: str) -> List[Dict[str, Any]]:
        """Encontra ancestrais de um conceito."""
        ancestors = []
        
        def find_parents(current_concept: str, depth: int = 0):
            if current_concept in self.concepts:
                concept = self.concepts[current_concept]
                for parent in concept.parent_concepts:
                    ancestors.append({
                        "name": parent,
                        "relationship": "parent",
                        "depth": depth + 1,
                        "abstraction_level": self.concepts[parent].abstraction_level.value
                    })
                    find_parents(parent, depth + 1)
        
        find_parents(concept_name)
        return ancestors
    
    def _find_descendants(self, concept_name: str) -> List[Dict[str, Any]]:
        """Encontra descendentes de um conceito."""
        descendants = []
        
        def find_children(current_concept: str, depth: int = 0):
            if current_concept in self.concepts:
                concept = self.concepts[current_concept]
                for child in concept.child_concepts:
                    descendants.append({
                        "name": child,
                        "relationship": "child",
                        "depth": depth + 1,
                        "abstraction_level": self.concepts[child].abstraction_level.value
                    })
                    find_children(child, depth + 1)
        
        find_children(concept_name)
        return descendants
    
    def _find_related_concepts(self, concept_name: str) -> List[Dict[str, Any]]:
        """Encontra conceitos relacionados."""
        related = []
        
        for relationship in self.relationships:
            if relationship.source_concept == concept_name:
                related.append({
                    "name": relationship.target_concept,
                    "relationship_type": relationship.relationship_type.value,
                    "strength": relationship.strength,
                    "confidence": relationship.confidence
                })
            elif relationship.target_concept == concept_name:
                related.append({
                    "name": relationship.source_concept,
                    "relationship_type": relationship.relationship_type.value,
                    "strength": relationship.strength,
                    "confidence": relationship.confidence
                })
        
        return related
    
    def _calculate_hierarchy_depth(self, concept_name: str) -> int:
        """Calcula a profundidade de um conceito na hierarquia."""
        if concept_name not in self.concepts:
            return 0
        
        concept = self.concepts[concept_name]
        if not concept.parent_concepts:
            return 0
        
        max_depth = 0
        for parent in concept.parent_concepts:
            depth = self._calculate_hierarchy_depth(parent) + 1
            max_depth = max(max_depth, depth)
        
        return max_depth
    
    def _calculate_concept_centrality(self, concept_name: str) -> float:
        """Calcula a centralidade de um conceito na hierarquia."""
        try:
            # Usar centralidade de intermediação do NetworkX
            centrality = nx.betweenness_centrality(self.concept_graph)
            return centrality.get(concept_name, 0.0)
        except:
            return 0.0
    
    def _get_inherited_characteristics(self, concept_name: str) -> List[str]:
        """Obtém características herdadas de um conceito."""
        inherited = []
        
        def collect_characteristics(current_concept: str):
            if current_concept in self.concepts:
                concept = self.concepts[current_concept]
                inherited.extend(concept.characteristics)
                for parent in concept.parent_concepts:
                    collect_characteristics(parent)
        
        collect_characteristics(concept_name)
        return list(set(inherited))  # Remover duplicatas
    
    def _analyze_examples(self, concept_name: str) -> Dict[str, Any]:
        """Analisa exemplos e contra-exemplos de um conceito."""
        if concept_name not in self.concepts:
            return {}
        
        concept = self.concepts[concept_name]
        
        return {
            "examples": concept.examples,
            "counter_examples": concept.counter_examples,
            "examples_count": len(concept.examples),
            "counter_examples_count": len(concept.counter_examples),
            "example_diversity": len(set(concept.examples)) / max(1, len(concept.examples))
        }
    
    def learn_from_example(self, concept_name: str, example: str, is_positive: bool = True, 
                          feedback: str = ""):
        """
        Aprende com um exemplo de um conceito.
        
        Args:
            concept_name: Nome do conceito
            example: Exemplo fornecido
            is_positive: Se é um exemplo positivo ou negativo
            feedback: Feedback adicional
        """
        if concept_name not in self.concepts:
            logger.warning(f"Conceito '{concept_name}' não encontrado")
            return
        
        concept = self.concepts[concept_name]
        
        if is_positive:
            if example not in concept.examples:
                concept.examples.append(example)
        else:
            if example not in concept.counter_examples:
                concept.counter_examples.append(example)
        
        # Atualizar frequência de uso
        concept.usage_frequency += 1
        concept.last_used = time.time()
        
        # Atualizar score de confiança baseado no feedback
        if feedback:
            if "excellent" in feedback.lower() or "perfect" in feedback.lower():
                concept.confidence_score = min(1.0, concept.confidence_score + 0.05)
            elif "good" in feedback.lower():
                concept.confidence_score = min(1.0, concept.confidence_score + 0.02)
            elif "bad" in feedback.lower() or "wrong" in feedback.lower():
                concept.confidence_score = max(0.0, concept.confidence_score - 0.02)
        
        logger.info(f"Exemplo aprendido para '{concept_name}': {example} ({'positivo' if is_positive else 'negativo'})")
    
    def find_similar_concepts(self, concept_name: str, threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Encontra conceitos similares usando análise de grafos.
        
        Args:
            concept_name: Nome do conceito base
            threshold: Threshold de similaridade
            
        Returns:
            Lista de conceitos similares
        """
        if concept_name not in self.concepts:
            return []
        
        similar_concepts = []
        
        # Usar o grafo de similaridade
        if concept_name in self.similarity_graph:
            for neighbor in self.similarity_graph.neighbors(concept_name):
                weight = self.similarity_graph[concept_name][neighbor]['weight']
                if weight >= threshold:
                    similar_concepts.append({
                        "name": neighbor,
                        "similarity": weight,
                        "abstraction_level": self.concepts[neighbor].abstraction_level.value
                    })
        
        # Ordenar por similaridade
        similar_concepts.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similar_concepts
    
    def infer_concept_properties(self, concept_name: str) -> Dict[str, Any]:
        """
        Infere propriedades de um conceito baseado na hierarquia.
        
        Args:
            concept_name: Nome do conceito
            
        Returns:
            Propriedades inferidas
        """
        if concept_name not in self.concepts:
            return {"error": f"Conceito '{concept_name}' não encontrado"}
        
        concept = self.concepts[concept_name]
        
        # Inferir características baseadas na hierarquia
        inferred_characteristics = self._get_inherited_characteristics(concept_name)
        
        # Inferir comportamentos baseados em relacionamentos
        inferred_behaviors = []
        for relationship in concept.relationships:
            if relationship.relationship_type == ConceptRelationshipType.CAUSES:
                inferred_behaviors.append(relationship.target_concept)
        
        # Inferir propriedades baseadas no nível de abstração
        abstraction_properties = self._infer_abstraction_properties(concept.abstraction_level)
        
        # Inferir complexidade baseada na estrutura
        complexity_properties = self._infer_complexity_properties(concept.complexity)
        
        return {
            "concept_name": concept_name,
            "inferred_characteristics": inferred_characteristics,
            "inferred_behaviors": inferred_behaviors,
            "abstraction_properties": abstraction_properties,
            "complexity_properties": complexity_properties,
            "inference_confidence": self._calculate_inference_confidence(concept_name)
        }
    
    def _infer_abstraction_properties(self, abstraction_level: ConceptAbstractionLevel) -> Dict[str, Any]:
        """Infere propriedades baseadas no nível de abstração."""
        properties = {
            ConceptAbstractionLevel.CONCRETE: {
                "specificity": "high",
                "observability": "high",
                "generalizability": "low"
            },
            ConceptAbstractionLevel.SPECIFIC: {
                "specificity": "high",
                "observability": "high",
                "generalizability": "medium"
            },
            ConceptAbstractionLevel.GENERAL: {
                "specificity": "medium",
                "observability": "medium",
                "generalizability": "high"
            },
            ConceptAbstractionLevel.ABSTRACT: {
                "specificity": "low",
                "observability": "low",
                "generalizability": "very_high"
            },
            ConceptAbstractionLevel.UNIVERSAL: {
                "specificity": "very_low",
                "observability": "very_low",
                "generalizability": "maximum"
            },
            ConceptAbstractionLevel.METAPHYSICAL: {
                "specificity": "minimal",
                "observability": "minimal",
                "generalizability": "maximum"
            }
        }
        
        return properties.get(abstraction_level, {})
    
    def _infer_complexity_properties(self, complexity: ConceptComplexity) -> Dict[str, Any]:
        """Infere propriedades baseadas na complexidade."""
        properties = {
            ConceptComplexity.SIMPLE: {
                "dimensionality": 1,
                "interdependencies": "none",
                "learning_difficulty": "easy"
            },
            ConceptComplexity.COMPOUND: {
                "dimensionality": 2,
                "interdependencies": "few",
                "learning_difficulty": "medium"
            },
            ConceptComplexity.COMPLEX: {
                "dimensionality": 3,
                "interdependencies": "many",
                "learning_difficulty": "hard"
            },
            ConceptComplexity.META: {
                "dimensionality": 4,
                "interdependencies": "maximum",
                "learning_difficulty": "very_hard"
            }
        }
        
        return properties.get(complexity, {})
    
    def _calculate_inference_confidence(self, concept_name: str) -> float:
        """Calcula a confiança na inferência de um conceito."""
        if concept_name not in self.concepts:
            return 0.0
        
        concept = self.concepts[concept_name]
        
        # Baseado no número de relacionamentos e exemplos
        relationship_confidence = min(1.0, len(concept.relationships) / 10.0)
        example_confidence = min(1.0, len(concept.examples) / 20.0)
        usage_confidence = min(1.0, concept.usage_frequency / 100.0)
        
        # Média ponderada
        total_confidence = (relationship_confidence * 0.4 + 
                           example_confidence * 0.3 + 
                           usage_confidence * 0.3)
        
        return total_confidence
    
    def get_hierarchy_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas da hierarquia de conceitos."""
        total_concepts = len(self.concepts)
        total_relationships = len(self.relationships)
        
        # Estatísticas por nível de abstração
        abstraction_stats = defaultdict(int)
        for concept in self.concepts.values():
            abstraction_stats[concept.abstraction_level.value] += 1
        
        # Estatísticas por complexidade
        complexity_stats = defaultdict(int)
        for concept in self.concepts.values():
            complexity_stats[concept.complexity.value] += 1
        
        # Estatísticas por tipo de relacionamento
        relationship_stats = defaultdict(int)
        for relationship in self.relationships:
            relationship_stats[relationship.relationship_type.value] += 1
        
        # Métricas do grafo
        graph_metrics = {}
        try:
            graph_metrics = {
                "nodes": self.concept_graph.number_of_nodes(),
                "edges": self.concept_graph.number_of_edges(),
                "density": nx.density(self.concept_graph),
                "average_clustering": nx.average_clustering(self.concept_graph.to_undirected())
            }
        except:
            graph_metrics = {"error": "Erro ao calcular métricas do grafo"}
        
        return {
            "total_concepts": total_concepts,
            "total_relationships": total_relationships,
            "abstraction_level_distribution": dict(abstraction_stats),
            "complexity_distribution": dict(complexity_stats),
            "relationship_type_distribution": dict(relationship_stats),
            "graph_metrics": graph_metrics,
            "average_confidence": np.mean([c.confidence_score for c in self.concepts.values()]),
            "most_used_concept": max(self.concepts.values(), key=lambda c: c.usage_frequency).name if self.concepts else None
        }
    
    def save_hierarchy(self, file_path: str = "data/concept_hierarchy.json"):
        """Salva a hierarquia de conceitos."""
        try:
            data = {
                "concepts": {},
                "relationships": [],
                "metadata": {
                    "created_at": time.time(),
                    "total_concepts": len(self.concepts),
                    "total_relationships": len(self.relationships)
                }
            }
            
            # Converter conceitos para formato serializável
            for name, concept in self.concepts.items():
                data["concepts"][name] = {
                    "name": concept.name,
                    "definition": concept.definition,
                    "abstraction_level": concept.abstraction_level.value,
                    "complexity": concept.complexity.value,
                    "parent_concepts": concept.parent_concepts,
                    "child_concepts": concept.child_concepts,
                    "characteristics": concept.characteristics,
                    "examples": concept.examples,
                    "counter_examples": concept.counter_examples,
                    "confidence_score": concept.confidence_score,
                    "usage_frequency": concept.usage_frequency,
                    "last_used": concept.last_used,
                    "created_at": concept.created_at
                }
            
            # Converter relacionamentos para formato serializável
            for relationship in self.relationships:
                data["relationships"].append({
                    "source_concept": relationship.source_concept,
                    "target_concept": relationship.target_concept,
                    "relationship_type": relationship.relationship_type.value,
                    "strength": relationship.strength,
                    "confidence": relationship.confidence,
                    "evidence": relationship.evidence,
                    "created_at": relationship.created_at,
                    "last_updated": relationship.last_updated
                })
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=4)
            
            logger.info(f"Hierarquia de conceitos salva em {file_path}")
            
        except Exception as e:
            logger.error(f"Erro ao salvar hierarquia: {e}")
    
    def load_hierarchy(self, file_path: str = "data/concept_hierarchy.json"):
        """Carrega a hierarquia de conceitos."""
        if not os.path.exists(file_path):
            logger.warning(f"Arquivo {file_path} não encontrado")
            return
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Carregar conceitos
            for name, concept_data in data["concepts"].items():
                concept = HierarchicalConcept(
                    name=concept_data["name"],
                    definition=concept_data["definition"],
                    abstraction_level=ConceptAbstractionLevel(concept_data["abstraction_level"]),
                    complexity=ConceptComplexity(concept_data["complexity"]),
                    parent_concepts=concept_data["parent_concepts"],
                    child_concepts=concept_data["child_concepts"],
                    characteristics=concept_data["characteristics"],
                    examples=concept_data["examples"],
                    counter_examples=concept_data["counter_examples"],
                    confidence_score=concept_data["confidence_score"],
                    usage_frequency=concept_data["usage_frequency"],
                    last_used=concept_data["last_used"],
                    created_at=concept_data["created_at"]
                )
                
                self.concepts[name] = concept
                self.concept_graph.add_node(name, concept=concept)
            
            # Reconstruir arestas do grafo
            for name, concept in self.concepts.items():
                for parent in concept.parent_concepts:
                    if parent in self.concepts:
                        self.concept_graph.add_edge(parent, name)
            
            # Carregar relacionamentos
            for rel_data in data["relationships"]:
                relationship = ConceptRelationship(
                    source_concept=rel_data["source_concept"],
                    target_concept=rel_data["target_concept"],
                    relationship_type=ConceptRelationshipType(rel_data["relationship_type"]),
                    strength=rel_data["strength"],
                    confidence=rel_data["confidence"],
                    evidence=rel_data["evidence"],
                    created_at=rel_data["created_at"],
                    last_updated=rel_data["last_updated"]
                )
                
                self.relationships.append(relationship)
                
                # Adicionar ao grafo de similaridade se aplicável
                if relationship.relationship_type == ConceptRelationshipType.SIMILAR_TO:
                    self.similarity_graph.add_edge(
                        relationship.source_concept, 
                        relationship.target_concept, 
                        weight=relationship.strength
                    )
            
            logger.info(f"Hierarquia de conceitos carregada: {len(self.concepts)} conceitos, {len(self.relationships)} relacionamentos")
            
        except Exception as e:
            logger.error(f"Erro ao carregar hierarquia: {e}")
