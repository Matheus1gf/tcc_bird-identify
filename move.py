import os
import shutil

base_dir = './data/images'
train_dir = './data/train'
validation_dir = './data/validation'

# Criar diretórios das classes dentro de train e validation
with open('./data/classes.txt', 'r') as f:
    classes = {line.split()[0]: line.split()[1] for line in f.readlines()}

for class_id in classes.keys():
    os.makedirs(os.path.join(train_dir, classes[class_id]), exist_ok=True)
    os.makedirs(os.path.join(validation_dir, classes[class_id]), exist_ok=True)

# Mover imagens para treino ou validação
with open('./data/train_test_split.txt', 'r') as f:
    for line in f:
        image_id, is_train = line.strip().split()
        is_train = int(is_train)
        
        # Obter o nome da imagem a partir do ID
        with open('./data/images.txt', 'r') as img_file:
            img_dict = {line.split()[0]: line.split()[1] for line in img_file.readlines()}
        
        image_name = img_dict[image_id]
        class_id = image_name.split('/')[0]

        # Origem e destino
        src = os.path.join(base_dir, image_name)
        dest = os.path.join(train_dir if is_train else validation_dir, class_id, os.path.basename(image_name))
        
        # Mover a imagem
        shutil.move(src, dest)

print("Organização concluída!")
