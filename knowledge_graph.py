import networkx as nx
import json
import logging
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)

class NodeType(Enum):
    SPECIE = "especie"
    ANATOMICAL_PART = "parte_anatomica"
    CHARACTERISTIC = "caracteristica"
    HABITAT = "habitat"
    BEHAVIOR = "comportamento"

class RelationType(Enum):
    HAS_PART = "tem_parte"
    SIMILAR_TO = "similar_a"
    FOUND_IN = "encontrado_em"
    EXHIBITS = "exibe"
    PREDICTS = "prediz"

@dataclass
class GraphNode:
    id: str
    node_type: NodeType
    properties: Dict
    confidence: float = 1.0

@dataclass
class GraphEdge:
    source: str
    target: str
    relation_type: RelationType
    weight: float = 1.0
    confidence: float = 1.0

class KnowledgeGraph:
    """
    Grafo de Conhecimento para o sistema de identificação de pássaros.
    Implementa memória de longo prazo com relações semânticas.
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_counter = 0
        self._initialize_base_knowledge()
    
    def _initialize_base_knowledge(self):
        """Inicializa conhecimento base sobre pássaros"""
        
        # Adicionar partes anatômicas conhecidas
        anatomical_parts = [
            ("bico", {"shape": "variável", "color": "variável", "size": "variável"}),
            ("asa", {"shape": "alongada", "function": "voo"}),
            ("olho", {"shape": "redondo", "function": "visao"}),
            ("garra", {"shape": "curva", "function": "agarrar"}),
            ("cauda", {"shape": "variável", "function": "equilibrio"}),
            ("corpo", {"shape": "ovoidal", "function": "protecao"})
        ]
        
        for part, props in anatomical_parts:
            self.add_node(f"part_{part}", NodeType.ANATOMICAL_PART, props)
        
        # Adicionar relações básicas
        self._add_base_relations()
    
    def _add_base_relations(self):
        """Adiciona relações básicas entre partes anatômicas"""
        
        # Todas as aves têm essas partes básicas
        basic_parts = ["bico", "asa", "olho", "garra", "cauda", "corpo"]
        
        for part in basic_parts:
            self.add_edge("especie_ave_generica", f"part_{part}", 
                         RelationType.HAS_PART, weight=1.0)
    
    def add_node(self, node_id: str, node_type: NodeType, 
                properties: Dict, confidence: float = 1.0) -> None:
        """Adiciona um nó ao grafo"""
        
        if self.graph.has_node(node_id):
            # Atualizar propriedades existentes
            current_props = self.graph.nodes[node_id].get('properties', {})
            current_props.update(properties)
            self.graph.nodes[node_id]['properties'] = current_props
        else:
            # Criar novo nó
            self.graph.add_node(node_id, 
                              node_type=node_type.value,
                              properties=properties,
                              confidence=confidence)
    
    def add_edge(self, source: str, target: str, 
                relation_type: RelationType, weight: float = 1.0,
                confidence: float = 1.0) -> None:
        """Adiciona uma aresta ao grafo"""
        
        self.graph.add_edge(source, target,
                          relation_type=relation_type.value,
                          weight=weight,
                          confidence=confidence)
    
    def add_species_from_analysis(self, species_name: str, 
                                detected_parts: List[str],
                                confidence: float) -> None:
        """Adiciona uma espécie baseada na análise do sistema"""
        
        species_id = f"especie_{species_name.replace(' ', '_').lower()}"
        
        # Adicionar nó da espécie
        self.add_node(species_id, NodeType.SPECIE, {
            "nome": species_name,
            "confianca_deteccao": confidence,
            "partes_detectadas": detected_parts
        }, confidence)
        
        # Adicionar relações com partes detectadas
        for part in detected_parts:
            part_id = f"part_{part}"
            if self.graph.has_node(part_id):
                self.add_edge(species_id, part_id, RelationType.HAS_PART, 
                            weight=confidence)
    
    def query_similar_species(self, detected_parts: List[str], 
                            threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Consulta espécies similares baseadas nas partes detectadas"""
        
        similar_species = []
        
        for node_id in self.graph.nodes():
            node_data = self.graph.nodes[node_id]
            
            if node_data.get('node_type') == NodeType.SPECIE.value:
                species_parts = node_data.get('properties', {}).get('partes_detectadas', [])
                
                # Calcular similaridade baseada em interseção de partes
                intersection = set(detected_parts) & set(species_parts)
                union = set(detected_parts) | set(species_parts)
                
                if union:
                    similarity = len(intersection) / len(union)
                    if similarity >= threshold:
                        species_name = node_data.get('properties', {}).get('nome', node_id)
                        similar_species.append((species_name, similarity))
        
        return sorted(similar_species, key=lambda x: x[1], reverse=True)
    
    def predict_missing_parts(self, species_name: str, 
                            detected_parts: List[str]) -> List[str]:
        """Prediz partes que podem estar presentes mas não foram detectadas"""
        
        species_id = f"especie_{species_name.replace(' ', '_').lower()}"
        
        if not self.graph.has_node(species_id):
            return []
        
        # Buscar espécies similares para inferir partes faltantes
        similar_species = self.query_similar_species(detected_parts)
        
        predicted_parts = []
        for similar_species_name, similarity in similar_species:
            similar_id = f"especie_{similar_species_name.replace(' ', '_').lower()}"
            
            if self.graph.has_node(similar_id):
                similar_parts = self.graph.nodes[similar_id].get('properties', {}).get('partes_detectadas', [])
                
                # Adicionar partes que não foram detectadas mas são comuns em espécies similares
                for part in similar_parts:
                    if part not in detected_parts and part not in predicted_parts:
                        predicted_parts.append(part)
        
        return predicted_parts
    
    def generate_species_blueprint(self, target_characteristics: Dict) -> Dict:
        """
        Gera um blueprint lógico para uma nova espécie baseado em características desejadas.
        Esta é a funcionalidade do Módulo de Inovação.
        """
        
        blueprint = {
            "especie_proposta": f"Nova_Especie_{self.node_counter}",
            "caracteristicas_desejadas": target_characteristics,
            "partes_necessarias": [],
            "partes_opcionais": [],
            "habitat_sugerido": [],
            "comportamentos_esperados": [],
            "confianca_blueprint": 0.0
        }
        
        # Lógica para gerar blueprint baseado nas características desejadas
        if "voo" in target_characteristics.get("habilidades", []):
            blueprint["partes_necessarias"].extend(["asa", "corpo"])
            blueprint["confianca_blueprint"] += 0.3
        
        if "caca" in target_characteristics.get("habilidades", []):
            blueprint["partes_necessarias"].extend(["bico", "garra"])
            blueprint["confianca_blueprint"] += 0.2
        
        if "navegacao" in target_characteristics.get("habilidades", []):
            blueprint["partes_necessarias"].extend(["olho", "cauda"])
            blueprint["confianca_blueprint"] += 0.2
        
        # Normalizar confiança
        blueprint["confianca_blueprint"] = min(blueprint["confianca_blueprint"], 1.0)
        
        return blueprint
    
    def get_graph_statistics(self) -> Dict:
        """Retorna estatísticas do grafo"""
        
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "species_count": len([n for n in self.graph.nodes() 
                                if self.graph.nodes[n].get('node_type') == NodeType.SPECIE.value]),
            "anatomical_parts_count": len([n for n in self.graph.nodes() 
                                         if self.graph.nodes[n].get('node_type') == NodeType.ANATOMICAL_PART.value]),
            "density": nx.density(self.graph),
            "is_connected": nx.is_weakly_connected(self.graph)
        }
    
    def save_graph(self, filepath: str) -> None:
        """Salva o grafo em formato JSON"""
        
        graph_data = {
            "nodes": [
                {
                    "id": node_id,
                    "type": data.get('node_type'),
                    "properties": data.get('properties', {}),
                    "confidence": data.get('confidence', 1.0)
                }
                for node_id, data in self.graph.nodes(data=True)
            ],
            "edges": [
                {
                    "source": source,
                    "target": target,
                    "relation_type": data.get('relation_type'),
                    "weight": data.get('weight', 1.0),
                    "confidence": data.get('confidence', 1.0)
                }
                for source, target, data in self.graph.edges(data=True)
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)
    
    def load_graph(self, filepath: str) -> None:
        """Carrega o grafo de um arquivo JSON"""
        
        with open(filepath, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)
        
        # Limpar grafo atual
        self.graph.clear()
        
        # Adicionar nós
        for node_data in graph_data.get("nodes", []):
            self.graph.add_node(node_data["id"],
                              node_type=node_data.get("type"),
                              properties=node_data.get("properties", {}),
                              confidence=node_data.get("confidence", 1.0))
        
        # Adicionar arestas
        for edge_data in graph_data.get("edges", []):
            self.graph.add_edge(edge_data["source"], edge_data["target"],
                              relation_type=edge_data.get("relation_type"),
                              weight=edge_data.get("weight", 1.0),
                              confidence=edge_data.get("confidence", 1.0))

# Exemplo de uso
if __name__ == "__main__":
    # Criar grafo de conhecimento
    kg = KnowledgeGraph()
    
    # Adicionar espécie detectada pelo sistema
    kg.add_species_from_analysis("Painted Bunting", ["bico", "asa"], 0.85)
    
    # Consultar espécies similares
    similar = kg.query_similar_species(["bico", "asa"])
    print("Espécies similares:", similar)
    
    # Gerar blueprint para nova espécie
    blueprint = kg.generate_species_blueprint({
        "habilidades": ["voo", "caca"],
        "habitat": "floresta"
    })
    print("Blueprint gerado:", blueprint)
    
    # Salvar grafo
    kg.save_graph("knowledge_graph.json")
    
    # Estatísticas
    stats = kg.get_graph_statistics()
    print("Estatísticas do grafo:", stats)
