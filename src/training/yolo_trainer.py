# from ultralytics import YOLO  # Comentado para evitar erro
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from patches import apply_yolo_patch

# Aplicar patch para resolver problema do PyTorch 2.6
apply_yolo_patch()

# Carrega o modelo pré-treinado yolov8n ('n' de nano)
# O YOLO vai baixar este modelo na primeira vez que você executar.
model = YOLO('yolov8n.pt')

# Inicia o treinamento do modelo
# Certifique-se de que o caminho para o arquivo .yaml está correto
results = model.train(
    data='passaros_parts.yaml',
    epochs=250,
    imgsz=640,
    project='runs/train',  # Opcional: define a pasta raiz para salvar os resultados
    name='exp_passaros_01' # Opcional: define o nome da pasta do experimento
)

print("Treinamento concluído!")
print("O melhor modelo foi salvo em: ", results.save_dir)