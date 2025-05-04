import cv2
import numpy as np
import os
import tensorflow as tf
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
# Limites HSV para cada característica
LIMITE_BICO = [
    (np.array([8, 150, 150]), np.array([40, 255, 255])),  # Amarelo/Laranja
    (np.array([0, 0, 0]), np.array([40, 40, 100]))        # Preto/Cinza claro
]
LIMITE_PENAS = [
    (np.array([0, 0, 50]), np.array([180, 80, 200])),  # Cores opacas
    (np.array([20, 80, 120]), np.array([40, 255, 255])),  # Amarelo vivo
    (np.array([0, 30, 50]), np.array([20, 180, 200])),     # Amarelo mais opaco
    (np.array([0, 40, 30]), np.array([15, 180, 200])),    # Vermelho pálido
    (np.array([90, 40, 50]), np.array([130, 255, 255])), # Tons azulados (penas escuras)
    (np.array([40, 40, 50]), np.array([80, 255, 255])),   # Tons esverdeados (penas exóticas)
    (np.array([0, 0, 0]), np.array([180, 255, 50])), # Tons de preto
    (np.array([0, 0, 200]), np.array([180, 20, 255])), # Tons de branco
    (np.array([10, 100, 100]), np.array([20, 255, 255])), # Tons de Marrom (marrom-avermelhado, marrom-escuro, marrom-claro, castanho)
    (np.array([130, 100, 100]), np.array([160, 255, 255])), # Tons deRoxo/Púrpura (roxo-escuro, roxo-azulado, púrpura-brilhante)
    (np.array([0, 0, 50]), np.array([180, 50, 200])), # Tons de Cinza (cinza-claro, cinza-escuro, cinza-prateado)
    (np.array([30, 100, 100]), np.array([40, 255, 255])), # Tons de Dourado (dourado-metálico, dourado-vibrante)
    (np.array([140, 100, 100]), np.array([170, 255, 255])), # Tons de Rosa (rosa-claro, rosa-escuro, rosa-pálido)
    (np.array([10, 20, 150]), np.array([30, 150, 255])), # Tons de Bege (bege-areia, bege-amarelado, bege-claro)
    (np.array([0, 0, 180]), np.array([180, 20, 255])) # Tons de Marfim (marfim-claro, marfim-escuro)
]
LIMITE_GARRAS = (np.array([0, 20, 20]), np.array([40, 150, 150]))  # Cinza/Bege
LIMITE_ASAS = [
    (np.array([0, 0, 0]), np.array([40, 50, 80])),   # Preto/Cinza escuro
    (np.array([10, 100, 20]), np.array([25, 180, 150]))  # Marrom Claro
]
LIMITE_PERNAS_ESCAMAS = [
    (np.array([15, 100, 100]), np.array([35, 255, 255])),  # Amarelo claro
    (np.array([0, 0, 50]), np.array([40, 40, 120]))        # Cinza claro
]
LIMITE_CAUDA = [
    (np.array([0, 0, 0]), np.array([30, 40, 70])),     # Preto/Cinza escuro
    (np.array([10, 90, 20]), np.array([30, 160, 130])) # Marrom
]
LIMITE_OLHOS = [
    (np.array([0, 0, 0]), np.array([30, 30, 50])),    # Preto profundo
    (np.array([0, 0, 50]), np.array([40, 50, 120]))  # Cinza escuro
]
LIMITE_POSTURA = (np.array([0, 0, 40]), np.array([180, 50, 200]))  # Tons neutros em área inferior
LIMITE_SEM_PELOS = (np.array([0, 0, 40]), np.array([180, 40, 180]))  # Superfície lisa
LIMITE_SEM_ORELHAS = (np.array([0, 0, 30]), np.array([180, 50, 150]))  # Laterais homogêneas

modelo_especies = tf.keras.models.load_model('modelo_classificacao_passaros.h5')

# Função de processamento de imagem
def processar_imagem(imagem, limite_inferior, limite_superior, tipo_filtro='open'):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    mascara = cv2.inRange(hsv, limite_inferior, limite_superior)
    mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
    kernel = np.ones((5, 5), np.uint8)
    if tipo_filtro == 'open':
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    else:
        mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    return cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

def identificar_especie(imagem):
    imagem = cv2.resize(imagem, (224, 224))  # Ajuste para o tamanho do modelo
    imagem = np.expand_dims(imagem, axis=0)  # Adiciona a dimensão do lote
    imagem = imagem / 255.0  # Normaliza a imagem

    predicao = modelo_especies.predict(imagem)
    print(f"Predição bruta: {predicao}")
    especie = np.argmax(predicao, axis=1)  # Obtém a classe com maior probabilidade
    return especie

def detectar_caracteristica(imagem, limites, area_min, area_max, vertices_min=3, vertices_max=15):
    for limite in limites if isinstance(limites, list) else [limites]:
        contornos = processar_imagem(imagem, *limite)
        for contorno in contornos:
            area = cv2.contourArea(contorno)
            if area_min <= area <= area_max:
                epsilon = 0.02 * cv2.arcLength(contorno, True)
                aprox = cv2.approxPolyDP(contorno, epsilon, True)
                if vertices_min <= len(aprox) <= vertices_max:
                    return True
    return False

def gerar_limite_hsv(imagem):
    hsv = cv2.cvtColor(imagem, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    h_min, h_max = np.min(h), np.max(h)
    s_min, s_max = np.min(s), np.max(s)
    v_min, v_max = np.min(v), np.max(v)
    return (np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))

class NivelCerteza(Enum):
    ALTA = "90% de chance"
    MEDIA = "80% de chance"
    BAIXA = "70% de chance"
    NAO_E = "Não é um pássaro"

# Carrega os nomes das classes em uma lista (índice 0 corresponde à classe 1 do arquivo)
def carregar_nomes_classes(caminho='data/classes.txt'):
    nomes = []
    with open(caminho, 'r') as f:
        for linha in f:
            partes = linha.strip().split()
            nome_classe = ' '.join(partes[1:]).replace('_', ' ')
            nomes.append(nome_classe)
    return nomes

def identificar_caracteristicas(caminho_imagem):
    imagem = cv2.imread(caminho_imagem)
    if imagem is None:
        logging.error(f"Erro ao carregar a imagem: {caminho_imagem}")
        return
    especie = identificar_especie(imagem)
    imagem = cv2.resize(imagem, (300, 300))

    encontrou_bico = detectar_caracteristica(imagem, LIMITE_BICO, 100, 3000, 5, 12)
    encontrou_penas = detectar_caracteristica(imagem, LIMITE_PENAS, 500, 20000, 8, 50) 
    encontrou_garras = detectar_caracteristica(imagem, LIMITE_GARRAS, 200, 3500, 3, 8)
    encontrou_asas = detectar_caracteristica(imagem, LIMITE_ASAS, 1000, 15000, 5, 30)
    encontrou_pernas = detectar_caracteristica(imagem, LIMITE_PERNAS_ESCAMAS, 200, 4000, 4, 10)
    encontrou_cauda = detectar_caracteristica(imagem, LIMITE_CAUDA, 1000, 12000, 5, 20)
    encontrou_olhos = detectar_caracteristica(imagem, LIMITE_OLHOS, 50, 800, 3, 6)
    encontrou_postura = detectar_caracteristica(imagem, LIMITE_POSTURA, 3000, 20000, 10, 40)
    sem_pelos = detectar_caracteristica(imagem, LIMITE_SEM_PELOS, 2000, 20000, 5, 30)
    sem_orelhas = detectar_caracteristica(imagem, LIMITE_SEM_ORELHAS, 500, 5000, 4, 15)

    nomes_classes = carregar_nomes_classes()
    nome_especie = nomes_classes[especie.item()]
    
    print(f"Resultado para {caminho_imagem}:")
    if (encontrou_bico and encontrou_penas and encontrou_garras and encontrou_asas and encontrou_pernas and encontrou_cauda and encontrou_olhos and encontrou_postura and sem_pelos and sem_orelhas):
        logging.info(f"{NivelCerteza.ALTA.value} de ser um pássaro.")
        print(f"Espécie identificada: {nome_especie}")
    elif(encontrou_bico and encontrou_penas and encontrou_garras and encontrou_asas and encontrou_olhos and encontrou_postura and (encontrou_cauda or sem_pelos or sem_orelhas)):
        logging.info(f"{NivelCerteza.MEDIA.value} de ser um pássaro.")
        print(f"Espécie identificada: {nome_especie}")
    elif(encontrou_bico and encontrou_penas and (encontrou_garras or encontrou_asas or encontrou_olhos or encontrou_postura or encontrou_cauda or sem_pelos or sem_orelhas)):
        logging.info(f"{NivelCerteza.BAIXA.value} de ser um pássaro.")
        print(f"Espécie identificada: {nome_especie}")
    else:
        logging.info(f"{NivelCerteza.NAO_E.value}")
        limite_hsv = gerar_limite_hsv(imagem)
        print(f'Limite HSV gerado: {limite_hsv}')
        
def processar_diretorio(diretorio):
    for arquivo in os.listdir(diretorio):
        if arquivo.endswith((".jpeg", ".jpg", ".png")):
            identificar_caracteristicas(os.path.join(diretorio, arquivo))

processar_diretorio('./dataset')
