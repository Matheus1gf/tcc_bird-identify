import json
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import networkx as nx

from knowledge_graph import KnowledgeGraph, NodeType, RelationType

logging.basicConfig(level=logging.INFO)

class InnovationGoal(Enum):
    """Tipos de metas de inovação"""
    OPTIMIZE_FLIGHT = "otimizar_voo"
    IMPROVE_CAMOUFLAGE = "melhorar_camuflagem"
    ENHANCE_HUNTING = "melhorar_caca"
    ADAPT_HABITAT = "adaptar_habitat"
    INCREASE_SPEED = "aumentar_velocidade"
    IMPROVE_NAVIGATION = "melhorar_navegacao"

class ConstraintType(Enum):
    """Tipos de restrições para blueprints"""
    PHYSICAL = "fisica"
    EVOLUTIONARY = "evolutiva"
    ECOLOGICAL = "ecologica"
    ENERGETIC = "energetica"

@dataclass
class InnovationConstraint:
    """Restrição para geração de blueprints"""
    constraint_type: ConstraintType
    description: str
    weight: float = 1.0
    is_hard_constraint: bool = False

@dataclass
class SpeciesBlueprint:
    """Blueprint de uma nova espécie"""
    name: str
    goal: InnovationGoal
    required_parts: List[str]
    optional_parts: List[str]
    part_modifications: Dict[str, Dict]
    habitat_preferences: List[str]
    behavioral_traits: List[str]
    physical_constraints: List[str]
    confidence_score: float
    feasibility_score: float
    innovation_score: float
    generation_timestamp: str = ""
    
    def __post_init__(self):
        if not self.generation_timestamp:
            self.generation_timestamp = datetime.now().isoformat()

class InnovationEngine:
    """
    Motor de inovação que gera blueprints lógicos de novas espécies
    baseado em metas definidas pelo usuário e conhecimento existente
    """
    
    def __init__(self, knowledge_graph: KnowledgeGraph):
        """
        Inicializa motor de inovação
        
        Args:
            knowledge_graph: Grafo de conhecimento com espécies existentes
        """
        self.knowledge_graph = knowledge_graph
        self.constraints = self._initialize_constraints()
        self.part_database = self._initialize_part_database()
    
    def _initialize_constraints(self) -> List[InnovationConstraint]:
        """Inicializa restrições para geração de blueprints"""
        
        constraints = [
            # Restrições físicas
            InnovationConstraint(
                ConstraintType.PHYSICAL,
                "Pássaros devem ter pelo menos bico e asas para sobrevivência básica",
                weight=1.0,
                is_hard_constraint=True
            ),
            InnovationConstraint(
                ConstraintType.PHYSICAL,
                "Tamanho do corpo deve ser proporcional ao tamanho das asas",
                weight=0.8
            ),
            InnovationConstraint(
                ConstraintType.PHYSICAL,
                "Olhos devem estar posicionados para visão binocular",
                weight=0.7
            ),
            
            # Restrições evolutivas
            InnovationConstraint(
                ConstraintType.EVOLUTIONARY,
                "Modificações devem ser evolutivamente plausíveis",
                weight=0.9
            ),
            InnovationConstraint(
                ConstraintType.EVOLUTIONARY,
                "Não pode haver conflitos entre características",
                weight=0.8
            ),
            
            # Restrições ecológicas
            InnovationConstraint(
                ConstraintType.ECOLOGICAL,
                "Habitat deve suportar as características propostas",
                weight=0.7
            ),
            InnovationConstraint(
                ConstraintType.ECOLOGICAL,
                "Deve haver nicho ecológico disponível",
                weight=0.6
            ),
            
            # Restrições energéticas
            InnovationConstraint(
                ConstraintType.ENERGETIC,
                "Características não podem consumir energia excessiva",
                weight=0.8
            ),
            InnovationConstraint(
                ConstraintType.ENERGETIC,
                "Vantagens devem superar custos energéticos",
                weight=0.7
            )
        ]
        
        return constraints
    
    def _initialize_part_database(self) -> Dict[str, Dict]:
        """Inicializa banco de dados de partes anatômicas e suas modificações"""
        
        return {
            "bico": {
                "base_function": "alimentacao",
                "modifications": {
                    "longo_curvo": {
                        "function": "pesca_profunda",
                        "energy_cost": 0.1,
                        "advantage": "alcance_maior",
                        "constraints": ["agua_profunda"]
                    },
                    "curto_robusto": {
                        "function": "quebra_sementes",
                        "energy_cost": 0.05,
                        "advantage": "forca_maior",
                        "constraints": ["sementes_duras"]
                    },
                    "afilado_agudo": {
                        "function": "caca_precisa",
                        "energy_cost": 0.15,
                        "advantage": "precisao_letal",
                        "constraints": ["presas_pequenas"]
                    }
                }
            },
            "asa": {
                "base_function": "voo",
                "modifications": {
                    "longa_estreita": {
                        "function": "voo_velocidade",
                        "energy_cost": 0.2,
                        "advantage": "velocidade_maxima",
                        "constraints": ["espacos_abertos"]
                    },
                    "curta_larga": {
                        "function": "manobrabilidade",
                        "energy_cost": 0.1,
                        "advantage": "agilidade",
                        "constraints": ["florestas_densas"]
                    },
                    "grande_arredondada": {
                        "function": "planagem",
                        "energy_cost": 0.05,
                        "advantage": "eficiencia_energetica",
                        "constraints": ["correntes_termicas"]
                    }
                }
            },
            "olho": {
                "base_function": "visao",
                "modifications": {
                    "grande_binocular": {
                        "function": "visao_precisa",
                        "energy_cost": 0.1,
                        "advantage": "profundidade_percepcao",
                        "constraints": ["caca_precisa"]
                    },
                    "pequeno_lateral": {
                        "function": "visao_ampliada",
                        "energy_cost": 0.05,
                        "advantage": "detecta_predadores",
                        "constraints": ["ambiente_perigoso"]
                    },
                    "adaptado_noturno": {
                        "function": "visao_baixa_luz",
                        "energy_cost": 0.15,
                        "advantage": "atividade_noturna",
                        "constraints": ["predadores_diurnos"]
                    }
                }
            },
            "garra": {
                "base_function": "agarrar",
                "modifications": {
                    "longa_curva": {
                        "function": "agarrar_presa",
                        "energy_cost": 0.1,
                        "advantage": "forca_agarre",
                        "constraints": ["presas_ativas"]
                    },
                    "curta_afilada": {
                        "function": "perfurar",
                        "energy_cost": 0.08,
                        "advantage": "penetracao",
                        "constraints": ["superficies_duras"]
                    },
                    "adaptada_escalada": {
                        "function": "escalar",
                        "energy_cost": 0.12,
                        "advantage": "mobilidade_vertical",
                        "constraints": ["superficies_rugosas"]
                    }
                }
            },
            "cauda": {
                "base_function": "equilibrio",
                "modifications": {
                    "longa_flexivel": {
                        "function": "manobrabilidade",
                        "energy_cost": 0.05,
                        "advantage": "controle_voo",
                        "constraints": ["manobras_complexas"]
                    },
                    "curta_rigida": {
                        "function": "estabilidade",
                        "energy_cost": 0.03,
                        "advantage": "voo_estavel",
                        "constraints": ["voo_retilíneo"]
                    },
                    "adaptada_sinalizacao": {
                        "function": "comunicacao",
                        "energy_cost": 0.08,
                        "advantage": "sinalizacao_visual",
                        "constraints": ["comportamento_social"]
                    }
                }
            },
            "corpo": {
                "base_function": "protecao",
                "modifications": {
                    "compacto_aerodinamico": {
                        "function": "velocidade",
                        "energy_cost": 0.1,
                        "advantage": "resistencia_ar",
                        "constraints": ["voo_velocidade"]
                    },
                    "robusto_protegido": {
                        "function": "resistencia",
                        "energy_cost": 0.15,
                        "advantage": "protecao_fisica",
                        "constraints": ["ambiente_hostil"]
                    },
                    "leve_eficiente": {
                        "function": "eficiencia_energetica",
                        "energy_cost": 0.05,
                        "advantage": "economia_energia",
                        "constraints": ["recursos_limitados"]
                    }
                }
            }
        }
    
    def generate_blueprint(self, goal: InnovationGoal, 
                          constraints: List[InnovationConstraint] = None,
                          target_habitat: str = None) -> SpeciesBlueprint:
        """
        Gera blueprint de nova espécie baseado em meta de inovação
        
        Args:
            goal: Meta de inovação
            constraints: Restrições adicionais
            target_habitat: Habitat alvo
            
        Returns:
            Blueprint da nova espécie
        """
        
        # Usar restrições padrão se não fornecidas
        if constraints is None:
            constraints = self.constraints
        
        # Gerar nome da espécie
        species_name = self._generate_species_name(goal)
        
        # Determinar partes necessárias e modificações
        required_parts, optional_parts, modifications = self._design_anatomy(
            goal, constraints, target_habitat
        )
        
        # Determinar habitat e comportamentos
        habitat_preferences = self._determine_habitat_preferences(goal, modifications)
        behavioral_traits = self._determine_behavioral_traits(goal, modifications)
        
        # Calcular scores
        confidence_score = self._calculate_confidence_score(modifications, constraints)
        feasibility_score = self._calculate_feasibility_score(modifications, constraints)
        innovation_score = self._calculate_innovation_score(goal, modifications)
        
        # Determinar restrições físicas
        physical_constraints = self._identify_physical_constraints(modifications)
        
        return SpeciesBlueprint(
            name=species_name,
            goal=goal,
            required_parts=required_parts,
            optional_parts=optional_parts,
            part_modifications=modifications,
            habitat_preferences=habitat_preferences,
            behavioral_traits=behavioral_traits,
            physical_constraints=physical_constraints,
            confidence_score=confidence_score,
            feasibility_score=feasibility_score,
            innovation_score=innovation_score
        )
    
    def _generate_species_name(self, goal: InnovationGoal) -> str:
        """Gera nome para nova espécie baseado na meta"""
        
        name_mapping = {
            InnovationGoal.OPTIMIZE_FLIGHT: "Velocidade Alada",
            InnovationGoal.IMPROVE_CAMOUFLAGE: "Mestre da Disfarce",
            InnovationGoal.ENHANCE_HUNTING: "Caçador Supremo",
            InnovationGoal.ADAPT_HABITAT: "Adaptador Universal",
            InnovationGoal.INCREASE_SPEED: "Raio Voador",
            InnovationGoal.IMPROVE_NAVIGATION: "Navegador Celestial"
        }
        
        base_name = name_mapping.get(goal, "Espécie Inovadora")
        timestamp = datetime.now().strftime("%Y%m%d")
        
        return f"{base_name}_{timestamp}"
    
    def _design_anatomy(self, goal: InnovationGoal, 
                       constraints: List[InnovationConstraint],
                       target_habitat: str) -> Tuple[List[str], List[str], Dict]:
        """Projeta anatomia baseada na meta"""
        
        required_parts = ["bico", "asa", "olho"]  # Mínimo para sobrevivência
        optional_parts = ["garra", "cauda", "corpo"]
        modifications = {}
        
        # Aplicar modificações baseadas na meta
        if goal == InnovationGoal.OPTIMIZE_FLIGHT:
            modifications["asa"] = "longa_estreita"
            modifications["corpo"] = "compacto_aerodinamico"
            modifications["cauda"] = "longa_flexivel"
            required_parts.extend(["corpo", "cauda"])
            
        elif goal == InnovationGoal.ENHANCE_HUNTING:
            modifications["bico"] = "afilado_agudo"
            modifications["garra"] = "longa_curva"
            modifications["olho"] = "grande_binocular"
            required_parts.extend(["garra"])
            
        elif goal == InnovationGoal.IMPROVE_CAMOUFLAGE:
            modifications["corpo"] = "adaptado_camuflagem"
            modifications["olho"] = "pequeno_lateral"
            modifications["cauda"] = "adaptada_sinalizacao"
            
        elif goal == InnovationGoal.IMPROVE_NAVIGATION:
            modifications["olho"] = "grande_binocular"
            modifications["asa"] = "grande_arredondada"
            modifications["corpo"] = "leve_eficiente"
            
        # Aplicar restrições
        modifications = self._apply_constraints(modifications, constraints)
        
        return required_parts, optional_parts, modifications
    
    def _apply_constraints(self, modifications: Dict, 
                          constraints: List[InnovationConstraint]) -> Dict:
        """Aplica restrições às modificações"""
        
        filtered_modifications = {}
        
        for part, modification in modifications.items():
            # Verificar se modificação é válida
            if part in self.part_database:
                part_data = self.part_database[part]
                if modification in part_data["modifications"]:
                    mod_data = part_data["modifications"][modification]
                    
                    # Verificar restrições físicas
                    if self._check_physical_constraints(mod_data, constraints):
                        filtered_modifications[part] = modification
        
        return filtered_modifications
    
    def _check_physical_constraints(self, mod_data: Dict, 
                                   constraints: List[InnovationConstraint]) -> bool:
        """Verifica se modificação atende restrições físicas"""
        
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.PHYSICAL:
                # Verificar custo energético
                if mod_data.get("energy_cost", 0) > 0.2:  # Limiar alto
                    if constraint.is_hard_constraint:
                        return False
        
        return True
    
    def _determine_habitat_preferences(self, goal: InnovationGoal, 
                                      modifications: Dict) -> List[str]:
        """Determina preferências de habitat baseadas na meta e modificações"""
        
        habitats = []
        
        if goal == InnovationGoal.OPTIMIZE_FLIGHT:
            habitats.extend(["espacos_abertos", "correntes_termicas"])
        
        if goal == InnovationGoal.ENHANCE_HUNTING:
            habitats.extend(["agua_profunda", "presas_ativas"])
        
        if goal == InnovationGoal.IMPROVE_CAMOUFLAGE:
            habitats.extend(["vegetacao_densa", "ambiente_variavel"])
        
        # Adicionar habitats baseados em modificações
        for part, modification in modifications.items():
            if part in self.part_database:
                mod_data = self.part_database[part]["modifications"].get(modification, {})
                habitats.extend(mod_data.get("constraints", []))
        
        return list(set(habitats))  # Remover duplicatas
    
    def _determine_behavioral_traits(self, goal: InnovationGoal, 
                                   modifications: Dict) -> List[str]:
        """Determina traços comportamentais baseados na meta"""
        
        traits = []
        
        if goal == InnovationGoal.OPTIMIZE_FLIGHT:
            traits.extend(["voo_velocidade", "manobras_complexas"])
        
        if goal == InnovationGoal.ENHANCE_HUNTING:
            traits.extend(["caca_precisa", "agressividade"])
        
        if goal == InnovationGoal.IMPROVE_CAMOUFLAGE:
            traits.extend(["comportamento_furtivo", "adaptacao_ambiente"])
        
        return traits
    
    def _calculate_confidence_score(self, modifications: Dict, 
                                   constraints: List[InnovationConstraint]) -> float:
        """Calcula score de confiança do blueprint"""
        
        # Baseado na plausibilidade evolutiva
        base_confidence = 0.7
        
        # Penalizar modificações muito complexas
        complexity_penalty = len(modifications) * 0.05
        
        # Bonificar modificações que atendem restrições
        constraint_bonus = 0.0
        for constraint in constraints:
            if constraint.constraint_type == ConstraintType.EVOLUTIONARY:
                constraint_bonus += 0.1
        
        confidence = base_confidence - complexity_penalty + constraint_bonus
        return max(0.0, min(1.0, confidence))
    
    def _calculate_feasibility_score(self, modifications: Dict, 
                                   constraints: List[InnovationConstraint]) -> float:
        """Calcula score de viabilidade do blueprint"""
        
        # Baseado em restrições físicas e energéticas
        base_feasibility = 0.8
        
        # Penalizar custos energéticos altos
        energy_penalty = 0.0
        for part, modification in modifications.items():
            if part in self.part_database:
                mod_data = self.part_database[part]["modifications"].get(modification, {})
                energy_cost = mod_data.get("energy_cost", 0)
                energy_penalty += energy_cost * 0.3
        
        feasibility = base_feasibility - energy_penalty
        return max(0.0, min(1.0, feasibility))
    
    def _calculate_innovation_score(self, goal: InnovationGoal, 
                                  modifications: Dict) -> float:
        """Calcula score de inovação do blueprint"""
        
        # Baseado na originalidade e potencial de impacto
        base_innovation = 0.6
        
        # Bonificar modificações únicas
        uniqueness_bonus = len(modifications) * 0.1
        
        # Bonificar metas ambiciosas
        goal_bonus = {
            InnovationGoal.OPTIMIZE_FLIGHT: 0.2,
            InnovationGoal.ENHANCE_HUNTING: 0.15,
            InnovationGoal.IMPROVE_CAMOUFLAGE: 0.1,
            InnovationGoal.ADAPT_HABITAT: 0.25,
            InnovationGoal.INCREASE_SPEED: 0.2,
            InnovationGoal.IMPROVE_NAVIGATION: 0.15
        }.get(goal, 0.1)
        
        innovation = base_innovation + uniqueness_bonus + goal_bonus
        return max(0.0, min(1.0, innovation))
    
    def _identify_physical_constraints(self, modifications: Dict) -> List[str]:
        """Identifica restrições físicas do blueprint"""
        
        constraints = []
        
        for part, modification in modifications.items():
            if part in self.part_database:
                mod_data = self.part_database[part]["modifications"].get(modification, {})
                constraints.extend(mod_data.get("constraints", []))
        
        return list(set(constraints))
    
    def compare_blueprints(self, blueprint1: SpeciesBlueprint, 
                          blueprint2: SpeciesBlueprint) -> Dict:
        """Compara dois blueprints e retorna análise"""
        
        comparison = {
            "blueprint1": blueprint1.name,
            "blueprint2": blueprint2.name,
            "confidence_comparison": {
                "blueprint1": blueprint1.confidence_score,
                "blueprint2": blueprint2.confidence_score,
                "winner": blueprint1.name if blueprint1.confidence_score > blueprint2.confidence_score else blueprint2.name
            },
            "feasibility_comparison": {
                "blueprint1": blueprint1.feasibility_score,
                "blueprint2": blueprint2.feasibility_score,
                "winner": blueprint1.name if blueprint1.feasibility_score > blueprint2.feasibility_score else blueprint2.name
            },
            "innovation_comparison": {
                "blueprint1": blueprint1.innovation_score,
                "blueprint2": blueprint2.innovation_score,
                "winner": blueprint1.name if blueprint1.innovation_score > blueprint2.innovation_score else blueprint2.name
            },
            "overall_recommendation": None
        }
        
        # Calcular recomendação geral
        score1 = (blueprint1.confidence_score + blueprint1.feasibility_score + blueprint1.innovation_score) / 3
        score2 = (blueprint2.confidence_score + blueprint2.feasibility_score + blueprint2.innovation_score) / 3
        
        comparison["overall_recommendation"] = blueprint1.name if score1 > score2 else blueprint2.name
        
        return comparison
    
    def save_blueprint(self, blueprint: SpeciesBlueprint, filepath: str):
        """Salva blueprint em arquivo JSON"""
        
        blueprint_dict = asdict(blueprint)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(blueprint_dict, f, indent=2, ensure_ascii=False)

# Exemplo de uso
if __name__ == "__main__":
    print("Módulo de Inovação implementado!")
    print("Para usar:")
    print("1. Inicialize InnovationEngine com KnowledgeGraph")
    print("2. Use generate_blueprint() com metas de inovação")
    print("3. Use compare_blueprints() para comparar propostas")
    print("4. Use save_blueprint() para salvar resultados")
