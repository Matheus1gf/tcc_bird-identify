import os
import json
import logging
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import requests
from dataclasses import dataclass, asdict
from pathlib import Path

from gradcam_module import AutoAnnotationSystem
from knowledge_graph import KnowledgeGraph

logging.basicConfig(level=logging.INFO)

@dataclass
class ValidationResult:
    """Resultado da validação externa"""
    is_valid: bool
    confidence: float
    suggested_class: Optional[str] = None
    feedback: str = ""
    api_response: Dict = None

@dataclass
class LearningSample:
    """Amostra de aprendizado"""
    image_path: str
    original_prediction: Dict
    gradcam_proposals: List[Dict]
    validation_result: Optional[ValidationResult]
    final_annotation: Optional[Dict] = None
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class ExternalValidator:
    """
    Validador externo usando APIs de visão (Gemini/GPT-4V)
    """
    
    def __init__(self, api_type: str = "gemini", api_key: str = None):
        """
        Inicializa validador externo
        
        Args:
            api_type: Tipo de API ("gemini" ou "gpt4v")
            api_key: Chave da API
        """
        self.api_type = api_type
        self.api_key = api_key
        self.base_url = self._get_api_url()
    
    def _get_api_url(self) -> str:
        """Retorna URL base da API"""
        if self.api_type == "gemini":
            return "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro-latest:generateContent"
        elif self.api_type == "gpt4v":
            return "https://api.openai.com/v1/chat/completions"
        else:
            raise ValueError(f"Tipo de API não suportado: {self.api_type}")
    
    def validate_annotation(self, image_path: str, 
                           proposed_annotation: Dict,
                           context: str = "") -> ValidationResult:
        """
        Valida uma anotação proposta usando API externa
        
        Args:
            image_path: Caminho para a imagem
            proposed_annotation: Anotação proposta pelo sistema
            context: Contexto adicional para validação
            
        Returns:
            Resultado da validação
        """
        try:
            if self.api_type == "gemini":
                return self._validate_with_gemini(image_path, proposed_annotation, context)
            elif self.api_type == "gpt4v":
                return self._validate_with_gpt4v(image_path, proposed_annotation, context)
        except Exception as e:
            logging.error(f"Erro na validação externa: {e}")
            return ValidationResult(
                is_valid=False,
                confidence=0.0,
                feedback=f"Erro na API: {str(e)}"
            )
    
    def _validate_with_gemini(self, image_path: str, 
                            proposed_annotation: Dict,
                            context: str) -> ValidationResult:
        """Validação usando Gemini API"""
        
        # Preparar imagem para API
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Preparar prompt
        prompt = f"""
        Analise esta imagem de pássaro e valide a seguinte anotação proposta:
        
        Anotação Proposta:
        - Classe: {proposed_annotation.get('class', 'N/A')}
        - Confiança: {proposed_annotation.get('confidence', 0):.2f}
        - Bounding Box: {proposed_annotation.get('bbox', [])}
        
        Contexto: {context}
        
        Por favor, responda em JSON com:
        {{
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "suggested_class": "nome_da_classe_correta",
            "feedback": "explicação_da_validação"
        }}
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
                "maxOutputTokens": 1024
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
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result['candidates'][0]['content']['parts'][0]['text']
            
            # Extrair JSON da resposta
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
                return ValidationResult(
                    is_valid=validation_data.get('is_valid', False),
                    confidence=validation_data.get('confidence', 0.0),
                    suggested_class=validation_data.get('suggested_class'),
                    feedback=validation_data.get('feedback', ''),
                    api_response=result
                )
        
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            feedback=f"Erro na API: Status {response.status_code}"
        )
    
    def _validate_with_gpt4v(self, image_path: str, 
                           proposed_annotation: Dict,
                           context: str) -> ValidationResult:
        """Validação usando GPT-4V API"""
        
        # Preparar imagem para API
        with open(image_path, 'rb') as f:
            image_data = f.read()
        
        import base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Preparar prompt
        prompt = f"""
        Analise esta imagem de pássaro e valide a seguinte anotação proposta:
        
        Anotação Proposta:
        - Classe: {proposed_annotation.get('class', 'N/A')}
        - Confiança: {proposed_annotation.get('confidence', 0):.2f}
        - Bounding Box: {proposed_annotation.get('bbox', [])}
        
        Contexto: {context}
        
        Responda em JSON:
        {{
            "is_valid": true/false,
            "confidence": 0.0-1.0,
            "suggested_class": "nome_da_classe_correta",
            "feedback": "explicação_da_validação"
        }}
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
            "max_tokens": 1024,
            "temperature": 0.1
        }
        
        # Fazer requisição
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        response = requests.post(self.base_url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Extrair JSON da resposta
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                validation_data = json.loads(json_match.group())
                return ValidationResult(
                    is_valid=validation_data.get('is_valid', False),
                    confidence=validation_data.get('confidence', 0.0),
                    suggested_class=validation_data.get('suggested_class'),
                    feedback=validation_data.get('feedback', ''),
                    api_response=result
                )
        
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            feedback=f"Erro na API: Status {response.status_code}"
        )

class ContinuousLearningSystem:
    """
    Sistema de aprendizado contínuo que implementa Human-in-the-Loop
    """
    
    def __init__(self, auto_annotation_system: AutoAnnotationSystem,
                 knowledge_graph: KnowledgeGraph,
                 external_validator: ExternalValidator,
                 learning_data_path: str = "./learning_data"):
        """
        Inicializa sistema de aprendizado contínuo
        
        Args:
            auto_annotation_system: Sistema de auto-anotação
            knowledge_graph: Grafo de conhecimento
            external_validator: Validador externo
            learning_data_path: Caminho para dados de aprendizado
        """
        self.auto_annotation_system = auto_annotation_system
        self.knowledge_graph = knowledge_graph
        self.external_validator = external_validator
        self.learning_data_path = Path(learning_data_path)
        
        # Criar diretórios necessários
        self.learning_data_path.mkdir(exist_ok=True)
        (self.learning_data_path / "pending_validation").mkdir(exist_ok=True)
        (self.learning_data_path / "validated").mkdir(exist_ok=True)
        (self.learning_data_path / "rejected").mkdir(exist_ok=True)
        
        # Carregar histórico de aprendizado
        self.learning_history = self._load_learning_history()
    
    def _load_learning_history(self) -> List[Dict]:
        """Carrega histórico de aprendizado"""
        history_file = self.learning_data_path / "learning_history.json"
        
        if history_file.exists():
            with open(history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return []
    
    def _save_learning_history(self):
        """Salva histórico de aprendizado"""
        history_file = self.learning_data_path / "learning_history.json"
        
        with open(history_file, 'w', encoding='utf-8') as f:
            json.dump(self.learning_history, f, indent=2, ensure_ascii=False)
    
    def process_new_image(self, image_path: str, 
                         auto_validate: bool = False) -> LearningSample:
        """
        Processa uma nova imagem para aprendizado contínuo
        
        Args:
            image_path: Caminho para a imagem
            auto_validate: Se deve validar automaticamente com API externa
            
        Returns:
            Amostra de aprendizado processada
        """
        # Analisar imagem com sistema de auto-anotação
        analysis = self.auto_annotation_system.analyze_image(image_path)
        
        # Criar amostra de aprendizado
        learning_sample = LearningSample(
            image_path=image_path,
            original_prediction=analysis["species_prediction"],
            gradcam_proposals=analysis["gradcam_proposals"],
            validation_result=None
        )
        
        # Validação externa se solicitada
        if auto_validate and analysis["needs_human_validation"]:
            validation_result = self._validate_with_external_api(learning_sample)
            learning_sample.validation_result = validation_result
        
        # Salvar amostra
        self._save_learning_sample(learning_sample)
        
        return learning_sample
    
    def _validate_with_external_api(self, learning_sample: LearningSample) -> ValidationResult:
        """Valida amostra com API externa"""
        
        # Usar a melhor proposta do Grad-CAM
        best_proposal = None
        if learning_sample.gradcam_proposals:
            best_proposal = max(learning_sample.gradcam_proposals, 
                              key=lambda x: x["confidence"])
        
        if best_proposal:
            context = f"Espécie predita: {learning_sample.original_prediction['species']}"
            return self.external_validator.validate_annotation(
                learning_sample.image_path,
                best_proposal,
                context
            )
        
        return ValidationResult(
            is_valid=False,
            confidence=0.0,
            feedback="Nenhuma proposta válida para validação"
        )
    
    def _save_learning_sample(self, sample: LearningSample):
        """Salva amostra de aprendizado"""
        
        # Converter para dicionário
        sample_dict = asdict(sample)
        
        # Salvar em arquivo individual
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"sample_{timestamp}.json"
        
        if sample.validation_result and sample.validation_result.is_valid:
            save_path = self.learning_data_path / "validated" / filename
        elif sample.validation_result and not sample.validation_result.is_valid:
            save_path = self.learning_data_path / "rejected" / filename
        else:
            save_path = self.learning_data_path / "pending_validation" / filename
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(sample_dict, f, indent=2, ensure_ascii=False)
        
        # Adicionar ao histórico
        self.learning_history.append(sample_dict)
        self._save_learning_history()
    
    def update_knowledge_graph(self, validated_samples: List[LearningSample]):
        """
        Atualiza grafo de conhecimento com amostras validadas
        
        Args:
            validated_samples: Amostras validadas pelo humano/API
        """
        for sample in validated_samples:
            if sample.validation_result and sample.validation_result.is_valid:
                # Extrair partes detectadas
                detected_parts = []
                for proposal in sample.gradcam_proposals:
                    if proposal["confidence"] > 0.5:
                        detected_parts.append(proposal.get("class", "unknown"))
                
                # Adicionar ao grafo de conhecimento
                species_name = sample.validation_result.suggested_class or \
                             sample.original_prediction["species"]
                
                self.knowledge_graph.add_species_from_analysis(
                    species_name,
                    detected_parts,
                    sample.validation_result.confidence
                )
        
        # Salvar grafo atualizado
        self.knowledge_graph.save_graph(
            str(self.learning_data_path / "updated_knowledge_graph.json")
        )
    
    def generate_learning_report(self) -> Dict:
        """Gera relatório de aprendizado contínuo"""
        
        total_samples = len(self.learning_history)
        validated_samples = len([s for s in self.learning_history 
                               if s.get('validation_result', {}).get('is_valid')])
        pending_samples = len([s for s in self.learning_history 
                             if not s.get('validation_result')])
        
        # Estatísticas do grafo de conhecimento
        graph_stats = self.knowledge_graph.get_graph_statistics()
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_samples_processed": total_samples,
            "validated_samples": validated_samples,
            "pending_validation": pending_samples,
            "validation_rate": validated_samples / total_samples if total_samples > 0 else 0,
            "knowledge_graph_stats": graph_stats,
            "recent_learning_samples": self.learning_history[-10:] if self.learning_history else []
        }
        
        return report
    
    def retrain_models(self, retrain_threshold: int = 50):
        """
        Retreina modelos quando há dados suficientes
        
        Args:
            retrain_threshold: Número mínimo de amostras validadas para retreinar
        """
        validated_count = len([s for s in self.learning_history 
                             if s.get('validation_result', {}).get('is_valid')])
        
        if validated_count >= retrain_threshold:
            logging.info(f"Iniciando retreinamento com {validated_count} amostras validadas")
            
            # Aqui você implementaria a lógica de retreinamento
            # Por exemplo, coletar dados validados e retreinar modelos
            
            # Por enquanto, apenas log
            logging.info("Retreinamento seria executado aqui")
            
            return True
        
        return False

# Exemplo de uso
if __name__ == "__main__":
    print("Sistema de Aprendizado Contínuo implementado!")
    print("Para usar:")
    print("1. Configure suas APIs externas (Gemini/GPT-4V)")
    print("2. Inicialize ContinuousLearningSystem")
    print("3. Use process_new_image() para processar novas imagens")
    print("4. Use update_knowledge_graph() para atualizar conhecimento")
    print("5. Use generate_learning_report() para relatórios")
