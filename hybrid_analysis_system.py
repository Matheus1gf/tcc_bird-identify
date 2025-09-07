#!/usr/bin/env python3
"""
Sistema Híbrido de Análise: APIs Externas + Análise Manual
Combina ChatGPT/Gemini com análise manual humana
"""

import os
import json
import requests
from typing import Dict, Any, Optional
from datetime import datetime
import base64
from PIL import Image
import io

class HybridAnalysisSystem:
    """Sistema híbrido que usa APIs externas quando disponíveis, senão análise manual"""
    
    def __init__(self):
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.use_apis = bool(self.gemini_api_key or self.openai_api_key)
        
    def analyze_image_with_api(self, image_path: str, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa imagem usando APIs externas"""
        
        if not self.use_apis:
            return {
                'status': 'no_apis',
                'message': 'APIs não configuradas, usando análise manual',
                'recommendation': 'manual_analysis'
            }
        
        try:
            # Tentar Gemini primeiro
            if self.gemini_api_key:
                result = self._analyze_with_gemini(image_path, detection_data)
                if result['status'] == 'success':
                    return result
            
            # Tentar OpenAI se Gemini falhar
            if self.openai_api_key:
                result = self._analyze_with_openai(image_path, detection_data)
                if result['status'] == 'success':
                    return result
            
            return {
                'status': 'api_error',
                'message': 'Todas as APIs falharam',
                'recommendation': 'manual_analysis'
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Erro na análise com API: {e}',
                'recommendation': 'manual_analysis'
            }
    
    def _analyze_with_gemini(self, image_path: str, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa imagem usando Gemini API"""
        
        try:
            # Codificar imagem em base64
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Preparar prompt
            prompt = self._create_analysis_prompt(detection_data)
            
            # Configurar requisição para Gemini
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={self.gemini_api_key}"
            
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt},
                            {
                                "inline_data": {
                                    "mime_type": "image/jpeg",
                                    "data": image_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "topK": 32,
                    "topP": 1,
                    "maxOutputTokens": 1024,
                }
            }
            
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' in result and result['candidates']:
                analysis_text = result['candidates'][0]['content']['parts'][0]['text']
                
                # Parse da resposta
                parsed_result = self._parse_api_response(analysis_text)
                parsed_result['api_used'] = 'gemini'
                parsed_result['status'] = 'success'
                
                return parsed_result
            else:
                return {
                    'status': 'error',
                    'message': 'Resposta inválida da API Gemini',
                    'recommendation': 'manual_analysis'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Erro na API Gemini: {e}',
                'recommendation': 'manual_analysis'
            }
    
    def _analyze_with_openai(self, image_path: str, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa imagem usando OpenAI Vision API"""
        
        try:
            # Codificar imagem em base64
            with open(image_path, 'rb') as image_file:
                image_data = base64.b64encode(image_file.read()).decode('utf-8')
            
            # Preparar prompt
            prompt = self._create_analysis_prompt(detection_data)
            
            # Configurar requisição para OpenAI
            url = "https://api.openai.com/v1/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "gpt-4o",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_data}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.1
            }
            
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            
            if 'choices' in result and result['choices']:
                analysis_text = result['choices'][0]['message']['content']
                
                # Parse da resposta
                parsed_result = self._parse_api_response(analysis_text)
                parsed_result['api_used'] = 'openai'
                parsed_result['status'] = 'success'
                
                return parsed_result
            else:
                return {
                    'status': 'error',
                    'message': 'Resposta inválida da API OpenAI',
                    'recommendation': 'manual_analysis'
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Erro na API OpenAI: {e}',
                'recommendation': 'manual_analysis'
            }
    
    def _create_analysis_prompt(self, detection_data: Dict[str, Any]) -> str:
        """Cria prompt para análise da imagem"""
        
        prompt = """
        Analise esta imagem de pássaro e responda APENAS com um JSON válido no seguinte formato:
        
        {
            "is_bird": true/false,
            "species": "nome_da_especie",
            "confidence": 0.0-1.0,
            "reasoning": ["motivo1", "motivo2"],
            "recommendation": "approve/reject/manual_review"
        }
        
        Critérios:
        - Se for claramente um pássaro: is_bird = true
        - Identifique a espécie se possível (bem_te_vi, sabia, beija_flor, etc.)
        - Confiança baseada na clareza da imagem
        - Se não tiver certeza: recommendation = "manual_review"
        
        Dados de detecção disponíveis:
        """
        
        if detection_data.get('yolo_detections'):
            prompt += f"Detecções YOLO: {detection_data['yolo_detections']}\n"
        else:
            prompt += "Nenhuma detecção YOLO\n"
        
        prompt += f"Tipo de análise: {detection_data.get('analysis_type', 'unknown')}\n"
        
        return prompt
    
    def _parse_api_response(self, response_text: str) -> Dict[str, Any]:
        """Parse da resposta da API"""
        
        try:
            # Tentar extrair JSON da resposta
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            
            if json_match:
                json_str = json_match.group()
                result = json.loads(json_str)
                
                # Validar campos obrigatórios
                required_fields = ['is_bird', 'species', 'confidence', 'reasoning', 'recommendation']
                for field in required_fields:
                    if field not in result:
                        result[field] = None
                
                return result
            else:
                return {
                    'is_bird': None,
                    'species': None,
                    'confidence': None,
                    'reasoning': ['Resposta da API não pôde ser parseada'],
                    'recommendation': 'manual_review'
                }
                
        except Exception as e:
            return {
                'is_bird': None,
                'species': None,
                'confidence': None,
                'reasoning': [f'Erro ao parsear resposta: {e}'],
                'recommendation': 'manual_review'
            }
    
    def get_analysis_recommendation(self, image_path: str, detection_data: Dict[str, Any]) -> Dict[str, Any]:
        """Obtém recomendação de análise (API ou manual)"""
        
        if not self.use_apis:
            return {
                'method': 'manual',
                'reason': 'APIs não configuradas',
                'recommendation': 'Usar análise manual'
            }
        
        # Analisar com API
        api_result = self.analyze_image_with_api(image_path, detection_data)
        
        if api_result['status'] == 'success':
            return {
                'method': 'api',
                'api_used': api_result.get('api_used', 'unknown'),
                'result': api_result,
                'recommendation': 'Análise automática concluída'
            }
        else:
            return {
                'method': 'manual',
                'reason': api_result.get('message', 'API falhou'),
                'recommendation': 'Usar análise manual'
            }

# Instância global do sistema híbrido
hybrid_analysis = HybridAnalysisSystem()
