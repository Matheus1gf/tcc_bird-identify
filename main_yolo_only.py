#!/usr/bin/env python3
"""
Sistema principal usando apenas YOLO (funciona agora!)
Baseado no main.py original, mas usando apenas detecção YOLO
"""

import cv2
import numpy as np
import os
import logging
import json
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

# --- CONFIGURAÇÃO ---
YOLO_MODEL_PATH = 'yolov8n.pt'  # Usar modelo pré-treinado
KNOWLEDGE_BASE_PATH = 'base_conhecimento.json'
CONFIDENCE_THRESHOLD = 0.60

# --- CARREGAMENTO DO MODELO ---
logging.info(f"Carregando modelo YOLO de '{YOLO_MODEL_PATH}'...")
modelo_yolo = YOLO(YOLO_MODEL_PATH)

base_conhecimento = {"especies_conhecidas": {}, "especies_novas": []}

def salvar_base_conhecimento(caminho=KNOWLEDGE_BASE_PATH):
    with open(caminho, 'w') as f:
        json.dump(base_conhecimento, f, indent=4, sort_keys=True)

def carregar_base_conhecimento(caminho=KNOWLEDGE_BASE_PATH):
    global base_conhecimento
    if os.path.exists(caminho):
        try:
            with open(caminho, 'r') as f:
                base_conhecimento = json.load(f)
                logging.info("Base de conhecimento carregada com sucesso.")
        except json.JSONDecodeError:
            logging.error("Erro ao decodificar base_conhecimento.json. Criando uma nova.")
            base_conhecimento = {"especies_conhecidas": {}, "especies_novas": []}
    else:
        logging.warning("Base de conhecimento não encontrada, criando uma nova.")

    # Garantir compatibilidade
    if 'especies_conhecidas' not in base_conhecimento:
        base_conhecimento['especies_conhecidas'] = {}
    if 'especies_novas' not in base_conhecimento:
        base_conhecimento['especies_novas'] = []

def extrair_fatos_com_yolo(imagem):
    """Extrai fatos visuais usando YOLO"""
    results = modelo_yolo(imagem, verbose=False)
    fatos_detectados = set()
    
    for r in results:
        for box in r.boxes:
            if box.conf > CONFIDENCE_THRESHOLD:  # Aplicar limiar
                class_id = int(box.cls)
                class_name = modelo_yolo.names[class_id]
                fatos_detectados.add(class_name)
    
    return fatos_detectados

def analisar_imagem_passaro(caminho_imagem):
    """Análise de imagem usando apenas YOLO"""
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        logging.error(f"Erro ao carregar a imagem: {caminho_imagem}")
        return

    print("-" * 50)
    print(f"Analisando Imagem: {os.path.basename(caminho_imagem)}")
    
    # 1. RACIOCÍNIO BASEADO EM FATOS: É um pássaro?
    fatos_visuais = extrair_fatos_com_yolo(imagem)
    
    # Verificar se é pássaro
    is_bird = 'bird' in fatos_visuais

    if not is_bird:
        print("CONCLUSÃO: Não foi possível confirmar que é um pássaro.")
        print(f"Fatos Detectados: {list(fatos_visuais) if fatos_visuais else 'Nenhum'}")
        print("Análise encerrada para esta imagem.")
        print("-" * 50)
        return

    # 2. Se é um pássaro, vamos analisar mais detalhadamente
    print(f"Fatos Visuais Detectados: {list(fatos_visuais)}")
    print("CONCLUSÃO PRELIMINAR: A imagem contém um pássaro.")

    # 3. Análise mais detalhada (simulada)
    if 'bird' in fatos_visuais:
        # Simular análise de espécie baseada em características visuais
        if len(fatos_visuais) > 1:  # Se detectou outros objetos além de pássaro
            print("ANÁLISE AVANÇADA: Pássaro detectado com outros elementos na imagem.")
            print("RECOMENDAÇÃO: Usar modelo de classificação específico para identificar espécie.")
        else:
            print("ANÁLISE BÁSICA: Pássaro detectado isoladamente.")
            print("RECOMENDAÇÃO: Imagem adequada para análise de espécie.")
        
        # Adicionar à base de conhecimento
        especie_temporaria = f"Pássaro_Detectado_{len(base_conhecimento['especies_conhecidas']) + 1}"
        base_conhecimento["especies_conhecidas"][especie_temporaria] = list(fatos_visuais)
        
        print(f"ESPÉCIE REGISTRADA: '{especie_temporaria}'")
        print("NOTA: Para identificação específica de espécie, treine o modelo de classificação.")

    salvar_base_conhecimento()
    print("-" * 50)

def processar_diretorio(diretorio):
    """Processa todas as imagens de um diretório"""
    for arquivo in os.listdir(diretorio):
        if arquivo.lower().endswith((".jpeg", ".jpg", ".png")):
            analisar_imagem_passaro(os.path.join(diretorio, arquivo))

# --- FLUXO DE EXECUÇÃO ---
if __name__ == "__main__":
    carregar_base_conhecimento()
    
    print("🐦 Sistema de Identificação de Pássaros (YOLO Only)")
    print("=" * 50)
    print("NOTA: Este sistema usa apenas detecção YOLO.")
    print("Para identificação específica de espécies, treine o modelo de classificação.")
    print("=" * 50)
    
    diretorio_de_teste = './dataset_teste'
    if not os.path.exists(diretorio_de_teste):
        print(f"\nAVISO: O diretório de teste '{diretorio_de_teste}' não foi encontrado.")
    else:
        processar_diretorio(diretorio_de_teste)
        
        print("\n📊 RESUMO DA ANÁLISE:")
        print(f"Espécies conhecidas: {len(base_conhecimento['especies_conhecidas'])}")
        print(f"Espécies novas: {len(base_conhecimento['especies_novas'])}")
