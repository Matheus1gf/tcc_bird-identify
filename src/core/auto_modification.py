"""
Sistema de Auto-Modificação de Código
=====================================

Este módulo implementa o sistema de auto-modificação de código para melhoria contínua.
Começando com a geração automática de novas regras baseadas em padrões aprendidos.

Funcionalidades:
- Geração automática de regras
- Análise de padrões para criar regras
- Validação de regras geradas
- Integração de regras ao sistema
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class RuleGenerator:
    """
    Gerador automático de novas regras baseado em padrões aprendidos.
    
    Este sistema analisa os padrões aprendidos pelo sistema e gera
    novas regras que podem melhorar a precisão e eficiência.
    """
    
    def __init__(self, learned_patterns_path: str = "data/learned_patterns.json"):
        self.learned_patterns_path = learned_patterns_path
        self.generated_rules_path = "data/generated_rules.json"
        self.rule_templates = self._load_rule_templates()
        self.learned_patterns = self._load_learned_patterns()
        
    def _load_rule_templates(self) -> Dict[str, Any]:
        """Carrega templates de regras para geração automática."""
        return {
            "color_pattern": {
                "template": "if color_dominant == '{color}' and confidence > {threshold}:",
                "description": "Regra baseada em cor dominante",
                "priority": "high"
            },
            "species_pattern": {
                "template": "if species_detected == '{species}' and characteristics_match:",
                "description": "Regra baseada em espécie detectada",
                "priority": "high"
            },
            "characteristic_pattern": {
                "template": "if has_{characteristic} and confidence > {threshold}:",
                "description": "Regra baseada em característica específica",
                "priority": "medium"
            },
            "combination_pattern": {
                "template": "if {condition1} and {condition2} and confidence > {threshold}:",
                "description": "Regra baseada em combinação de características",
                "priority": "high"
            },
            "confidence_boost": {
                "template": "confidence_boost += {boost_value}",
                "description": "Regra para aumentar confiança",
                "priority": "medium"
            }
        }
    
    def _load_learned_patterns(self) -> Dict[str, Any]:
        """Carrega padrões aprendidos do sistema."""
        try:
            if os.path.exists(self.learned_patterns_path):
                with open(self.learned_patterns_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Erro ao carregar padrões aprendidos: {e}")
            return {}
    
    def analyze_patterns(self) -> List[Dict[str, Any]]:
        """
        Analisa padrões aprendidos para identificar oportunidades de novas regras.
        
        Returns:
            Lista de regras sugeridas baseadas nos padrões
        """
        suggested_rules = []
        
        # Analisar padrões de cores
        color_rules = self._analyze_color_patterns()
        suggested_rules.extend(color_rules)
        
        # Analisar padrões de espécies
        species_rules = self._analyze_species_patterns()
        suggested_rules.extend(species_rules)
        
        # Analisar padrões de características
        characteristic_rules = self._analyze_characteristic_patterns()
        suggested_rules.extend(characteristic_rules)
        
        # Analisar combinações
        combination_rules = self._analyze_combination_patterns()
        suggested_rules.extend(combination_rules)
        
        return suggested_rules
    
    def _analyze_color_patterns(self) -> List[Dict[str, Any]]:
        """Analisa padrões de cores para gerar regras."""
        rules = []
        
        if "color_combinations" in self.learned_patterns:
            color_data = self.learned_patterns["color_combinations"]
            
            for color, data in color_data.items():
                if isinstance(data, dict) and "confidence" in data:
                    confidence = data["confidence"]
                    count = data.get("count", 0)
                    
                    # Gerar regra se a cor tem alta confiança e frequência
                    if confidence > 0.8 and count > 5:
                        rule = {
                            "type": "color_pattern",
                            "rule": f"if color_dominant == '{color}' and confidence > 0.7:",
                            "action": "confidence_boost += 0.15",
                            "description": f"Regra para cor {color} com alta confiança",
                            "priority": "high",
                            "generated_at": datetime.now().isoformat(),
                            "source": "color_analysis",
                            "confidence": confidence,
                            "frequency": count
                        }
                        rules.append(rule)
        
        return rules
    
    def _analyze_species_patterns(self) -> List[Dict[str, Any]]:
        """Analisa padrões de espécies para gerar regras."""
        rules = []
        
        if "known_species" in self.learned_patterns:
            species_data = self.learned_patterns["known_species"]
            
            for species, data in species_data.items():
                if isinstance(data, dict) and "confidence" in data:
                    confidence = data["confidence"]
                    count = data.get("count", 0)
                    
                    # Gerar regra se a espécie tem alta confiança
                    if confidence > 0.8 and count > 3:
                        rule = {
                            "type": "species_pattern",
                            "rule": f"if species_detected == '{species}' and characteristics_match:",
                            "action": "confidence_boost += 0.20",
                            "description": f"Regra para espécie {species}",
                            "priority": "high",
                            "generated_at": datetime.now().isoformat(),
                            "source": "species_analysis",
                            "confidence": confidence,
                            "frequency": count
                        }
                        rules.append(rule)
        
        return rules
    
    def _analyze_characteristic_patterns(self) -> List[Dict[str, Any]]:
        """Analisa padrões de características para gerar reglas."""
        rules = []
        
        if "characteristic_patterns" in self.learned_patterns:
            char_data = self.learned_patterns["characteristic_patterns"]
            
            for characteristic, data in char_data.items():
                if isinstance(data, dict) and "confidence" in data:
                    confidence = data["confidence"]
                    count = data.get("count", 0)
                    
                    # Gerar regra se a característica tem alta confiança
                    if confidence > 0.7 and count > 4:
                        rule = {
                            "type": "characteristic_pattern",
                            "rule": f"if has_{characteristic} and confidence > 0.6:",
                            "action": "confidence_boost += 0.10",
                            "description": f"Regra para característica {characteristic}",
                            "priority": "medium",
                            "generated_at": datetime.now().isoformat(),
                            "source": "characteristic_analysis",
                            "confidence": confidence,
                            "frequency": count
                        }
                        rules.append(rule)
        
        return rules
    
    def _analyze_combination_patterns(self) -> List[Dict[str, Any]]:
        """Analisa combinações de padrões para gerar regras complexas."""
        rules = []
        
        # Analisar combinações de cor + característica
        if "color_combinations" in self.learned_patterns and "characteristic_patterns" in self.learned_patterns:
            color_data = self.learned_patterns["color_combinations"]
            char_data = self.learned_patterns["characteristic_patterns"]
            
            for color, color_info in color_data.items():
                if isinstance(color_info, dict) and color_info.get("confidence", 0) > 0.7:
                    for char, char_info in char_data.items():
                        if isinstance(char_info, dict) and char_info.get("confidence", 0) > 0.7:
                            # Gerar regra de combinação
                            rule = {
                                "type": "combination_pattern",
                                "rule": f"if color_dominant == '{color}' and has_{char} and confidence > 0.6:",
                                "action": "confidence_boost += 0.25",
                                "description": f"Regra para combinação {color} + {char}",
                                "priority": "high",
                                "generated_at": datetime.now().isoformat(),
                                "source": "combination_analysis",
                                "confidence": (color_info["confidence"] + char_info["confidence"]) / 2,
                                "frequency": min(color_info.get("count", 0), char_info.get("count", 0))
                            }
                            rules.append(rule)
        
        return rules
    
    def generate_rule(self, pattern_type: str, parameters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Gera uma regra específica baseada no tipo e parâmetros.
        
        Args:
            pattern_type: Tipo de padrão (color_pattern, species_pattern, etc.)
            parameters: Parâmetros para a regra
            
        Returns:
            Regra gerada ou None se não for possível gerar
        """
        if pattern_type not in self.rule_templates:
            logger.warning(f"Tipo de padrão não reconhecido: {pattern_type}")
            return None
        
        template = self.rule_templates[pattern_type]
        
        try:
            # Substituir placeholders no template
            rule_text = template["template"].format(**parameters)
            
            rule = {
                "type": pattern_type,
                "rule": rule_text,
                "action": parameters.get("action", "confidence_boost += 0.1"),
                "description": template["description"],
                "priority": template["priority"],
                "generated_at": datetime.now().isoformat(),
                "source": "manual_generation",
                "parameters": parameters
            }
            
            return rule
            
        except KeyError as e:
            logger.error(f"Parâmetro faltando para gerar regra: {e}")
            return None
    
    def save_generated_rules(self, rules: List[Dict[str, Any]]) -> bool:
        """
        Salva regras geradas em arquivo.
        
        Args:
            rules: Lista de regras geradas
            
        Returns:
            True se salvou com sucesso, False caso contrário
        """
        try:
            # Carregar regras existentes
            existing_rules = []
            if os.path.exists(self.generated_rules_path):
                with open(self.generated_rules_path, 'r', encoding='utf-8') as f:
                    existing_rules = json.load(f)
            
            # Adicionar novas regras
            existing_rules.extend(rules)
            
            # Salvar regras atualizadas
            with open(self.generated_rules_path, 'w', encoding='utf-8') as f:
                json.dump(existing_rules, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Salvas {len(rules)} novas regras geradas")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao salvar regras geradas: {e}")
            return False
    
    def load_generated_rules(self) -> List[Dict[str, Any]]:
        """Carrega regras geradas do arquivo."""
        try:
            if os.path.exists(self.generated_rules_path):
                with open(self.generated_rules_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Erro ao carregar regras geradas: {e}")
            return []


class RuleValidator:
    """
    Validador de regras geradas para garantir que são válidas e seguras.
    """
    
    def __init__(self):
        self.valid_operators = ["==", "!=", ">", "<", ">=", "<=", "and", "or"]
        self.valid_actions = ["confidence_boost", "threshold_adjust", "feature_weight"]
        self.max_boost_value = 0.5  # Valor máximo para boost de confiança
    
    def validate_rule(self, rule: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Valida uma regra gerada.
        
        Args:
            rule: Regra a ser validada
            
        Returns:
            Tupla (é_válida, lista_de_erros)
        """
        errors = []
        
        # Validar estrutura básica
        required_fields = ["type", "rule", "action", "description"]
        for field in required_fields:
            if field not in rule:
                errors.append(f"Campo obrigatório '{field}' não encontrado")
        
        # Validar regra
        if "rule" in rule:
            rule_valid, rule_errors = self._validate_rule_syntax(rule["rule"])
            if not rule_valid:
                errors.extend(rule_errors)
        
        # Validar ação
        if "action" in rule:
            action_valid, action_errors = self._validate_action(rule["action"])
            if not action_valid:
                errors.extend(action_errors)
        
        # Validar prioridade
        if "priority" in rule:
            if rule["priority"] not in ["low", "medium", "high"]:
                errors.append("Prioridade deve ser 'low', 'medium' ou 'high'")
        
        return len(errors) == 0, errors
    
    def _validate_rule_syntax(self, rule_text: str) -> Tuple[bool, List[str]]:
        """Valida a sintaxe da regra."""
        errors = []
        
        # Verificar se contém palavras-chave válidas
        if not any(keyword in rule_text for keyword in ["if", "and", "or"]):
            errors.append("Regra deve conter 'if', 'and' ou 'or'")
        
        # Verificar operadores válidos
        for operator in self.valid_operators:
            if operator in rule_text:
                break
        else:
            errors.append("Regra deve conter pelo menos um operador válido")
        
        # Verificar se não contém código malicioso
        dangerous_patterns = ["import", "exec", "eval", "__", "os.", "sys."]
        for pattern in dangerous_patterns:
            if pattern in rule_text.lower():
                errors.append(f"Padrão perigoso detectado: {pattern}")
        
        return len(errors) == 0, errors
    
    def _validate_action(self, action: str) -> Tuple[bool, List[str]]:
        """Valida a ação da regra."""
        errors = []
        
        # Verificar se é uma ação válida
        if not any(valid_action in action for valid_action in self.valid_actions):
            errors.append("Ação deve conter uma operação válida")
        
        # Verificar valores de boost
        if "confidence_boost" in action:
            # Extrair valor do boost
            boost_match = re.search(r'confidence_boost\s*\+\=\s*([0-9.]+)', action)
            if boost_match:
                boost_value = float(boost_match.group(1))
                if boost_value > self.max_boost_value:
                    errors.append(f"Valor de boost muito alto: {boost_value} (máximo: {self.max_boost_value})")
        
        return len(errors) == 0, errors


class AutoModificationSystem:
    """
    Sistema principal de auto-modificação de código.
    
    Coordena a geração, validação e integração de regras automáticas.
    """
    
    def __init__(self):
        self.rule_generator = RuleGenerator()
        self.rule_validator = RuleValidator()
        self.generated_rules = []
        self.active_rules = []
        
    def run_auto_modification_cycle(self) -> Dict[str, Any]:
        """
        Executa um ciclo completo de auto-modificação.
        
        Returns:
            Relatório do ciclo executado
        """
        logger.info("Iniciando ciclo de auto-modificação")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "rules_generated": 0,
            "rules_validated": 0,
            "rules_integrated": 0,
            "errors": []
        }
        
        try:
            # 1. Analisar padrões e gerar regras
            logger.info("Analisando padrões aprendidos...")
            suggested_rules = self.rule_generator.analyze_patterns()
            report["rules_generated"] = len(suggested_rules)
            
            # 2. Validar regras geradas
            logger.info("Validando regras geradas...")
            valid_rules = []
            for rule in suggested_rules:
                is_valid, errors = self.rule_validator.validate_rule(rule)
                if is_valid:
                    valid_rules.append(rule)
                    report["rules_validated"] += 1
                else:
                    report["errors"].extend(errors)
                    logger.warning(f"Regra inválida: {errors}")
            
            # 3. Salvar regras válidas
            if valid_rules:
                self.rule_generator.save_generated_rules(valid_rules)
                self.generated_rules.extend(valid_rules)
                report["rules_integrated"] = len(valid_rules)
                logger.info(f"Integradas {len(valid_rules)} novas regras")
            
            # 4. Atualizar regras ativas
            self.active_rules = self.rule_generator.load_generated_rules()
            
        except Exception as e:
            error_msg = f"Erro no ciclo de auto-modificação: {e}"
            logger.error(error_msg)
            report["errors"].append(error_msg)
        
        logger.info(f"Ciclo de auto-modificação concluído: {report}")
        return report
    
    def get_system_status(self) -> Dict[str, Any]:
        """Retorna o status atual do sistema de auto-modificação."""
        return {
            "total_generated_rules": len(self.generated_rules),
            "active_rules": len(self.active_rules),
            "last_modification": datetime.now().isoformat(),
            "system_ready": True
        }


# Função de conveniência para uso externo
def create_auto_modification_system() -> AutoModificationSystem:
    """Cria e retorna uma instância do sistema de auto-modificação."""
    return AutoModificationSystem()
