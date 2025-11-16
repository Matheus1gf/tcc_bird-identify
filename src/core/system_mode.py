#!/usr/bin/env python3
"""
Sistema de Contexto de Uso
MELHORIA 4: Sistema que se adapta ao contexto de uso (pesquisa vs produção)
"""

import json
import os
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class SystemMode(Enum):
    """Modos de operação do sistema"""
    RESEARCH = "research"  # Pesquisa: mais permissivo, aceita mais casos
    PRODUCTION = "production"  # Produção: mais rigoroso, alta precisão
    BALANCED = "balanced"  # Balanceado: equilíbrio entre precisão e recall

class SystemModeManager:
    """Gerenciador de modos de operação do sistema"""
    
    def __init__(self, config_file: str = "config/system_mode.json"):
        """
        Inicializa gerenciador de modos
        
        Args:
            config_file: Caminho do arquivo de configuração
        """
        self.config_file = config_file
        self.config = self._load_config()
        
        # Determinar modo atual (variável de ambiente tem prioridade)
        mode_from_env = os.getenv('BIRD_IDENTIFY_MODE', '').lower()
        if mode_from_env in ['research', 'production', 'balanced']:
            self.current_mode = SystemMode(mode_from_env)
            logger.info(f"[SYSTEM_MODE] Modo definido por variável de ambiente: {self.current_mode.value}")
        else:
            self.current_mode = SystemMode(self.config.get('current_mode', 'balanced'))
            logger.info(f"[SYSTEM_MODE] Modo carregado do arquivo de configuração: {self.current_mode.value}")
        
        # Thresholds por modo
        self.mode_thresholds = self._get_mode_thresholds()
        
        # Estatísticas por modo
        self.mode_stats = {
            'research': {'total_analyses': 0, 'birds_detected': 0, 'false_positives': 0, 'false_negatives': 0},
            'production': {'total_analyses': 0, 'birds_detected': 0, 'false_positives': 0, 'false_negatives': 0},
            'balanced': {'total_analyses': 0, 'birds_detected': 0, 'false_positives': 0, 'false_negatives': 0}
        }
        
        # Carregar estatísticas salvas
        self._load_statistics()
    
    def _load_config(self) -> Dict[str, Any]:
        """Carrega configuração do arquivo"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"[SYSTEM_MODE] Erro ao carregar configuração: {e}")
        
        # Configuração padrão
        return {
            'current_mode': 'balanced',
            'min_confidence_thresholds': {
                'research': 0.3,
                'production': 0.7,
                'balanced': 0.5
            },
            'bird_like_features_thresholds': {
                'research': 0.2,
                'production': 0.5,
                'balanced': 0.35
            },
            'bird_shape_score_thresholds': {
                'research': 0.15,
                'production': 0.4,
                'balanced': 0.25
            },
            'bird_color_score_thresholds': {
                'research': 0.15,
                'production': 0.4,
                'balanced': 0.25
            }
        }
    
    def _save_config(self):
        """Salva configuração no arquivo"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            config_to_save = {
                'current_mode': self.current_mode.value,
                'last_updated': datetime.now().isoformat(),
                'min_confidence_thresholds': self.config.get('min_confidence_thresholds', {}),
                'bird_like_features_thresholds': self.config.get('bird_like_features_thresholds', {}),
                'bird_shape_score_thresholds': self.config.get('bird_shape_score_thresholds', {}),
                'bird_color_score_thresholds': self.config.get('bird_color_score_thresholds', {})
            }
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[SYSTEM_MODE] Erro ao salvar configuração: {e}")
    
    def _get_mode_thresholds(self) -> Dict[str, float]:
        """
        Retorna thresholds para o modo atual
        
        Returns:
            Dicionário com thresholds
        """
        mode_key = self.current_mode.value
        
        return {
            'min_confidence': self.config.get('min_confidence_thresholds', {}).get(mode_key, 0.5),
            'bird_like_features': self.config.get('bird_like_features_thresholds', {}).get(mode_key, 0.35),
            'bird_shape_score': self.config.get('bird_shape_score_thresholds', {}).get(mode_key, 0.25),
            'bird_color_score': self.config.get('bird_color_score_thresholds', {}).get(mode_key, 0.25)
        }
    
    def get_current_mode(self) -> SystemMode:
        """
        Retorna modo atual
        
        Returns:
            Modo de operação atual
        """
        return self.current_mode
    
    def set_mode(self, mode: SystemMode):
        """
        Define modo de operação
        
        Args:
            mode: Novo modo de operação
        """
        self.current_mode = mode
        self.mode_thresholds = self._get_mode_thresholds()
        self._save_config()
        logger.info(f"[SYSTEM_MODE] Modo alterado para: {mode.value}")
    
    def get_thresholds(self) -> Dict[str, float]:
        """
        Retorna thresholds para o modo atual
        
        Returns:
            Dicionário com thresholds
        """
        return self.mode_thresholds.copy()
    
    def get_min_confidence(self) -> float:
        """
        Retorna confiança mínima para o modo atual
        
        Returns:
            Confiança mínima (0.0 a 1.0)
        """
        return self.mode_thresholds['min_confidence']
    
    def should_accept_result(self, confidence: float, is_bird: bool) -> bool:
        """
        Determina se um resultado deve ser aceito baseado no modo atual
        
        Args:
            confidence: Confiança da predição
            is_bird: Se foi identificado como pássaro
        
        Returns:
            True se deve aceitar, False caso contrário
        """
        min_confidence = self.get_min_confidence()
        
        # Se confiança está abaixo do mínimo, rejeitar
        if confidence < min_confidence:
            return False
        
        # Em modo produção, ser mais rigoroso
        if self.current_mode == SystemMode.PRODUCTION:
            # Exigir confiança mais alta para pássaros
            if is_bird and confidence < 0.7:
                return False
        
        # Em modo pesquisa, ser mais permissivo
        if self.current_mode == SystemMode.RESEARCH:
            # Aceitar mesmo com confiança baixa para pesquisa
            return True
        
        # Modo balanceado: aceitar se confiança >= mínimo
        return confidence >= min_confidence
    
    def record_analysis(self, is_bird: bool, confidence: float, actual_bird: Optional[bool] = None):
        """
        Registra uma análise para estatísticas
        
        Args:
            is_bird: Se foi identificado como pássaro
            confidence: Confiança da predição
            actual_bird: Se realmente é pássaro (para calcular falsos positivos/negativos)
        """
        mode_key = self.current_mode.value
        stats = self.mode_stats[mode_key]
        
        stats['total_analyses'] += 1
        
        if is_bird:
            stats['birds_detected'] += 1
        
        # Calcular falsos positivos/negativos se tiver feedback
        if actual_bird is not None:
            if is_bird and not actual_bird:
                stats['false_positives'] += 1
            elif not is_bird and actual_bird:
                stats['false_negatives'] += 1
        
        # Salvar estatísticas periodicamente
        if stats['total_analyses'] % 10 == 0:
            self._save_statistics()
    
    def _load_statistics(self):
        """Carrega estatísticas do arquivo"""
        try:
            stats_file = self.config_file.replace('.json', '_stats.json')
            if os.path.exists(stats_file):
                with open(stats_file, 'r', encoding='utf-8') as f:
                    loaded_stats = json.load(f)
                    # Mesclar com estatísticas padrão
                    for mode in self.mode_stats.keys():
                        if mode in loaded_stats:
                            self.mode_stats[mode].update(loaded_stats[mode])
        except Exception as e:
            logger.error(f"[SYSTEM_MODE] Erro ao carregar estatísticas: {e}")
    
    def _save_statistics(self):
        """Salva estatísticas no arquivo"""
        try:
            stats_file = self.config_file.replace('.json', '_stats.json')
            os.makedirs(os.path.dirname(stats_file), exist_ok=True)
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'last_updated': datetime.now().isoformat(),
                    'statistics': self.mode_stats
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[SYSTEM_MODE] Erro ao salvar estatísticas: {e}")
    
    def get_statistics(self, mode: Optional[SystemMode] = None) -> Dict[str, Any]:
        """
        Retorna estatísticas para um modo
        
        Args:
            mode: Modo para obter estatísticas (None = modo atual)
        
        Returns:
            Dicionário com estatísticas
        """
        if mode is None:
            mode = self.current_mode
        
        mode_key = mode.value
        stats = self.mode_stats[mode_key]
        
        total = stats['total_analyses']
        if total == 0:
            return {
                'mode': mode_key,
                'total_analyses': 0,
                'birds_detected': 0,
                'detection_rate': 0.0,
                'false_positive_rate': 0.0,
                'false_negative_rate': 0.0,
                'accuracy': 0.0,
                'false_positives': stats.get('false_positives', 0),
                'false_negatives': stats.get('false_negatives', 0)
            }
        
        detection_rate = stats['birds_detected'] / total if total > 0 else 0.0
        false_positive_rate = stats['false_positives'] / total if total > 0 else 0.0
        false_negative_rate = stats['false_negatives'] / total if total > 0 else 0.0
        
        # Calcular precisão (se tiver feedback suficiente)
        correct = total - stats['false_positives'] - stats['false_negatives']
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'mode': mode_key,
            'total_analyses': total,
            'birds_detected': stats['birds_detected'],
            'detection_rate': detection_rate,
            'false_positive_rate': false_positive_rate,
            'false_negative_rate': false_negative_rate,
            'accuracy': accuracy,
            'false_positives': stats['false_positives'],
            'false_negatives': stats['false_negatives']
        }
    
    def get_all_statistics(self) -> Dict[str, Dict[str, Any]]:
        """
        Retorna estatísticas de todos os modos
        
        Returns:
            Dicionário com estatísticas de todos os modos
        """
        return {
            'research': self.get_statistics(SystemMode.RESEARCH),
            'production': self.get_statistics(SystemMode.PRODUCTION),
            'balanced': self.get_statistics(SystemMode.BALANCED)
        }
