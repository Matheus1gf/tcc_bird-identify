#!/usr/bin/env python3
"""
Curador Híbrido - O Supervisor Inteligente
Usa APIs de visão (Gemini/GPT-4V) para validar semanticamente
anotações geradas automaticamente, automatizando decisões de aprovação.
"""

import os
import json
import logging
import requests
import base64
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np

from auto_annotator import AutoAnnotation
from intuition_module import LearningCandidate

logging.basicConfig(level=logging.INFO)

class ValidationDecision(Enum):
    """Decisões de validação"""
    AUTO_APPROVE = "auto_approve"
    AUTO_REJECT = "auto_reject"
    HUMAN_REVIEW = "human_review"
    NEEDS_CORRECTION = "needs_correction"

class ValidationResult(Enum):
    """Resultados de validação"""
    YES_BIRD = "yes_bird"
    NO_BIRD = "no_bird"
    UNCERTAIN = "uncertain"
    ERROR = "error"

@dataclass
class ValidationResponse:
    """Resposta da validação via API"""
    result: ValidationResult
    confidence: float
    description: str
    reasoning: str
    api_response: Dict
    timestamp: str = ""

@dataclass
class CuratorDecision:
    """Decisão do curador híbrido"""
    decision: ValidationDecision
    confidence: float
    reasoning: str
    auto_annotation: AutoAnnotation
    validation_response: ValidationResponse
    human_review_needed: bool
    timestamp: str = ""

class HybridCurator:
    """
    Curador híbrido que usa APIs de visão para validação semântica
    """
    
    def __init__(self, api_type: str = "gemini", api_key: str = None):
        """
        Inicializa curador híbrido
        
        Args:
            api_type: Tipo de API ("gemini" ou "gpt4v")
            api_key: Chave da API
        """
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = self._get_api_url()
        
        # Configurações de validação
        self.grad_cam_strong_threshold = 0.6
        self.grad_cam_weak_threshold = 0.3
        self.keras_confidence_threshold = 0.4
        
        # Histórico de decisões
        self.curator_decisions = []
        
        # Estatísticas
        self.stats = {
            "total_validations": 0,
            "auto_approved": 0,
            "auto_rejected": 0,
            "human_review": 0,
            "api_errors": 0
        }
    
    def _get_api_url(self) -> str:
        """Retorna URL base da API"""
        if self.api_type == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
        elif self.api_type == "gpt4v":
            return "https://api.openai.com/v1/chat/completions"
        else:
            raise ValueError(f"Tipo de API não suportado: {self.api_type}")
    
    def validate_annotation(self, auto_annotation: AutoAnnotation) -> CuratorDecision:
        """
        Valida anotação usando API de visão
        
        Args:
            auto_annotation: Anotação gerada automaticamente
            
        Returns:
            Decisão do curador
        """
        self.stats["total_validations"] += 1
        
        try:
            # 1. Validação semântica via API
            validation_response = self._validate_with_api(auto_annotation.image_path)
            
            # 2. Raciocínio automatizado
            decision = self._make_automated_decision(auto_annotation, validation_response)
            
            # 3. Criar decisão do curador
            curator_decision = CuratorDecision(
                decision=decision["decision"],
                confidence=decision["confidence"],
                reasoning=decision["reasoning"],
                auto_annotation=auto_annotation,
                validation_response=validation_response,
                human_review_needed=decision["human_review_needed"],
                timestamp=datetime.now().isoformat()
            )
            
            # 4. Atualizar estatísticas
            self._update_stats(curator_decision)
            
            # 5. Adicionar ao histórico
            self.curator_decisions.append(curator_decision)
            
            logging.info(f"✅ Decisão do curador: {decision['decision'].value}")
            return curator_decision
            
        except Exception as e:
            logging.error(f"Erro na validação: {e}")
            self.stats["api_errors"] += 1
            
            # Decisão de fallback
            return CuratorDecision(
                decision=ValidationDecision.HUMAN_REVIEW,
                confidence=0.0,
                reasoning=f"Erro na validação: {str(e)}",
                auto_annotation=auto_annotation,
                validation_response=ValidationResponse(
                    result=ValidationResult.ERROR,
                    confidence=0.0,
                    description="Erro na API",
                    reasoning=str(e),
                    api_response={}
                ),
                human_review_needed=True,
                timestamp=datetime.now().isoformat()
            )
    
    def _validate_with_api(self, image_path: str) -> ValidationResponse:
        """
        Valida imagem usando API de visão
        
        Args:
            image_path: Caminho da imagem
            
        Returns:
            Resposta da validação
        """
        if self.api_type == "gemini":
            return self._validate_with_gemini(image_path)
        elif self.api_type == "gpt4v":
            return self._validate_with_gpt4v(image_path)
        else:
            raise ValueError(f"Tipo de API não suportado: {self.api_type}")
    
    def _validate_with_gemini(self, image_path: str) -> ValidationResponse:
        """Validação usando Gemini API"""
        
        # Preparar imagem
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prompt otimizado para validação
        prompt = """
        Analise esta imagem e responda EXATAMENTE com uma das opções:
        
        1. "SIM" - se o objeto principal é claramente um pássaro
        2. "NÃO" - se não é um pássaro ou é muito ambíguo
        3. "INCERTO" - se não consegue determinar com certeza
        
        Seja rigoroso: só responda "SIM" se tiver certeza absoluta de que é um pássaro.
        Responda apenas com uma palavra: SIM, NÃO ou INCERTO.
        """
        
        # Preparar payload
        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "image/jpeg",
                            "data": image_base64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.1,
                "topK": 1,
                "topP": 1,
                "maxOutputTokens": 10
            }
        }
        
        # Fazer requisição
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }
        
        response = requests.post(
            f"{self.base_url}?key={self.api_key}",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['candidates'][0]['content']['parts'][0]['text'].strip().upper()
            
            # Interpretar resposta
            if "SIM" in content:
                return ValidationResponse(
                    result=ValidationResult.YES_BIRD,
                    confidence=0.9,
                    description="Pássaro confirmado pela API",
                    reasoning="API respondeu SIM",
                    api_response=result
                )
            elif "NÃO" in content:
                return ValidationResponse(
                    result=ValidationResult.NO_BIRD,
                    confidence=0.9,
                    description="Não é pássaro segundo a API",
                    reasoning="API respondeu NÃO",
                    api_response=result
                )
            else:
                return ValidationResponse(
                    result=ValidationResult.UNCERTAIN,
                    confidence=0.5,
                    description="API incerta sobre a classificação",
                    reasoning="API respondeu INCERTO",
                    api_response=result
                )
        else:
            raise Exception(f"Erro na API Gemini: Status {response.status_code}")
    
    def _validate_with_gpt4v(self, image_path: str) -> ValidationResponse:
        """Validação usando GPT-4V API"""
        
        # Preparar imagem
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Prompt otimizado
        prompt = """
        Analise esta imagem e responda EXATAMENTE com uma das opções:
        
        1. "YES" - se o objeto principal é claramente um pássaro
        2. "NO" - se não é um pássaro ou é muito ambíguo
        3. "UNCERTAIN" - se não consegue determinar com certeza
        
        Seja rigoroso: só responda "YES" se tiver certeza absoluta de que é um pássaro.
        Responda apenas com uma palavra: YES, NO ou UNCERTAIN.
        """
        
        # Preparar payload
        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 10,
            "temperature": 0.1
        }
        
        # Fazer requisição
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content'].strip().upper()
            
            # Interpretar resposta
            if "YES" in content:
                return ValidationResponse(
                    result=ValidationResult.YES_BIRD,
                    confidence=0.9,
                    description="Pássaro confirmado pela API",
                    reasoning="API respondeu YES",
                    api_response=result
                )
            elif "NO" in content:
                return ValidationResponse(
                    result=ValidationResult.NO_BIRD,
                    confidence=0.9,
                    description="Não é pássaro segundo a API",
                    reasoning="API respondeu NO",
                    api_response=result
                )
            else:
                return ValidationResponse(
                    result=ValidationResult.UNCERTAIN,
                    confidence=0.5,
                    description="API incerta sobre a classificação",
                    reasoning="API respondeu UNCERTAIN",
                    api_response=result
                )
        else:
            raise Exception(f"Erro na API GPT-4V: Status {response.status_code}")
    
    def _make_automated_decision(self, auto_annotation: AutoAnnotation, 
                               validation_response: ValidationResponse) -> Dict:
        """
        Toma decisão automatizada baseada em múltiplas fontes
        
        Args:
            auto_annotation: Anotação gerada
            validation_response: Resposta da API
            
        Returns:
            Decisão automatizada
        """
        # Extrair métricas
        keras_confidence = auto_annotation.confidence
        grad_cam_strength = auto_annotation.grad_cam_strength
        api_result = validation_response.result
        api_confidence = validation_response.confidence
        
        # CENÁRIO 1: AUTO-APROVAÇÃO
        if (api_result == ValidationResult.YES_BIRD and 
            grad_cam_strength > self.grad_cam_strong_threshold and
            keras_confidence > self.keras_confidence_threshold):
            
            return {
                "decision": ValidationDecision.AUTO_APPROVE,
                "confidence": (api_confidence + grad_cam_strength + keras_confidence) / 3,
                "reasoning": f"✅ AUTO-APROVAÇÃO: API confirma pássaro, Grad-CAM forte ({grad_cam_strength:.2f}), Keras confiante ({keras_confidence:.2f})",
                "human_review_needed": False
            }
        
        # CENÁRIO 2: AUTO-REJEIÇÃO
        elif api_result == ValidationResult.NO_BIRD:
            return {
                "decision": ValidationDecision.AUTO_REJECT,
                "confidence": api_confidence,
                "reasoning": f"❌ AUTO-REJEIÇÃO: API confirma que não é pássaro",
                "human_review_needed": False
            }
        
        # CENÁRIO 3: NECESSITA REVISÃO HUMANA
        elif (api_result == ValidationResult.YES_BIRD and 
              grad_cam_strength < self.grad_cam_weak_threshold):
            return {
                "decision": ValidationDecision.HUMAN_REVIEW,
                "confidence": (api_confidence + keras_confidence) / 2,
                "reasoning": f"⚠️ REVISÃO HUMANA: API confirma pássaro, mas Grad-CAM fraco ({grad_cam_strength:.2f}) - bounding box pode ser impreciso",
                "human_review_needed": True
            }
        
        # CENÁRIO 4: INCERTEZA DA API
        elif api_result == ValidationResult.UNCERTAIN:
            return {
                "decision": ValidationDecision.HUMAN_REVIEW,
                "confidence": 0.5,
                "reasoning": f"❓ REVISÃO HUMANA: API incerta sobre a classificação",
                "human_review_needed": True
            }
        
        # CENÁRIO 5: CONFLITO DE EVIDÊNCIAS
        else:
            return {
                "decision": ValidationDecision.HUMAN_REVIEW,
                "confidence": (api_confidence + grad_cam_strength + keras_confidence) / 3,
                "reasoning": f"🔄 REVISÃO HUMANA: Conflito de evidências - API: {api_result.value}, Grad-CAM: {grad_cam_strength:.2f}, Keras: {keras_confidence:.2f}",
                "human_review_needed": True
            }
    
    def _update_stats(self, decision: CuratorDecision):
        """Atualiza estatísticas do curador"""
        if decision.decision == ValidationDecision.AUTO_APPROVE:
            self.stats["auto_approved"] += 1
        elif decision.decision == ValidationDecision.AUTO_REJECT:
            self.stats["auto_rejected"] += 1
        elif decision.decision == ValidationDecision.HUMAN_REVIEW:
            self.stats["human_review"] += 1
    
    def execute_decision(self, decision: CuratorDecision, 
                       train_dir: str = "./dataset_passaros/images/train",
                       awaiting_review_dir: str = "./awaiting_human_review",
                       rejected_dir: str = "./rejected_annotations") -> bool:
        """
        Executa a decisão do curador
        
        Args:
            decision: Decisão do curador
            train_dir: Diretório de treinamento
            awaiting_review_dir: Diretório para revisão humana
            rejected_dir: Diretório para anotações rejeitadas
            
        Returns:
            True se executado com sucesso
        """
        try:
            # Criar diretórios se não existirem
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(awaiting_review_dir, exist_ok=True)
            os.makedirs(rejected_dir, exist_ok=True)
            
            annotation = decision.auto_annotation
            image_path = annotation.image_path
            annotation_path = annotation.annotation_file_path
            
            # Determinar destino baseado na decisão
            if decision.decision == ValidationDecision.AUTO_APPROVE:
                # Mover para dataset de treinamento
                dest_dir = train_dir
                action = "APROVADO AUTOMATICAMENTE"
                
            elif decision.decision == ValidationDecision.AUTO_REJECT:
                # Mover para rejeitados
                dest_dir = rejected_dir
                action = "REJEITADO AUTOMATICAMENTE"
                
            else:  # HUMAN_REVIEW
                # Mover para revisão humana
                dest_dir = awaiting_review_dir
                action = "ENVIADO PARA REVISÃO HUMANA"
            
            # Copiar arquivos
            image_name = os.path.basename(image_path)
            annotation_name = os.path.basename(annotation_path)
            
            dest_image = os.path.join(dest_dir, image_name)
            dest_annotation = os.path.join(dest_dir, annotation_name)
            
            # Copiar imagem
            import shutil
            shutil.copy2(image_path, dest_image)
            
            # Copiar anotação
            shutil.copy2(annotation_path, dest_annotation)
            
            # Criar arquivo de decisão
            decision_file = os.path.join(dest_dir, f"{os.path.splitext(image_name)[0]}_decision.json")
            with open(decision_file, 'w') as f:
                json.dump(asdict(decision), f, indent=2, ensure_ascii=False)
            
            logging.info(f"✅ {action}: {image_name} -> {dest_dir}")
            return True
            
        except Exception as e:
            logging.error(f"Erro ao executar decisão: {e}")
            return False
    
    def get_curator_statistics(self) -> Dict:
        """Retorna estatísticas do curador"""
        stats = self.stats.copy()
        
        if stats["total_validations"] > 0:
            stats["auto_approval_rate"] = stats["auto_approved"] / stats["total_validations"]
            stats["auto_rejection_rate"] = stats["auto_rejected"] / stats["total_validations"]
            stats["human_review_rate"] = stats["human_review"] / stats["total_validations"]
            stats["api_error_rate"] = stats["api_errors"] / stats["total_validations"]
        else:
            stats.update({
                "auto_approval_rate": 0,
                "auto_rejection_rate": 0,
                "human_review_rate": 0,
                "api_error_rate": 0
            })
        
        return stats
    
    def generate_curator_report(self) -> Dict:
        """Gera relatório completo do curador"""
        return {
            "timestamp": datetime.now().isoformat(),
            "statistics": self.get_curator_statistics(),
            "recent_decisions": [
                {
                    "image": os.path.basename(d.auto_annotation.image_path),
                    "decision": d.decision.value,
                    "confidence": d.confidence,
                    "reasoning": d.reasoning
                }
                for d in self.curator_decisions[-10:]  # Últimas 10 decisões
            ],
            "efficiency_metrics": {
                "human_workload_reduction": f"{self.stats['auto_approved'] + self.stats['auto_rejected']} de {self.stats['total_validations']} casos automatizados",
                "accuracy_estimate": "Baseado na confiança média das decisões"
            }
        }

# Exemplo de uso
if __name__ == "__main__":
    print("🎭 Curador Híbrido - O Supervisor Inteligente")
    print("=" * 50)
    print("Este módulo usa APIs de visão para validar semanticamente")
    print("anotações geradas automaticamente, automatizando decisões.")
    print()
    print("Para usar:")
    print("1. Configure API_KEY_GEMINI ou API_KEY_GPT4V")
    print("2. Use validate_annotation() para validar anotações")
    print("3. Use execute_decision() para executar decisões")
    print()
    print("🚀 PRÓXIMO PASSO: Implementar Ciclo de Aprendizagem Contínua")
