# import tensorflow as tf  # Comentado para evitar erro
import numpy as np
# from tensorflow.keras  # Comentado para evitar erro.preprocessing.image import ImageDataGenerator
# from tensorflow.keras  # Comentado para evitar erro import layers, models
# from tensorflow.keras  # Comentado para evitar erro.applications.mobilenet_v2 import preprocess_input
# from sklearn.  # Comentado para evitar erroutils import class_weight
# from sklearn.  # Comentado para evitar erroensemble import RandomForestClassifier
# from sklearn.  # Comentado para evitar errosvm import SVC
# from skimage.  # Comentado para evitar errofeature import hog
from skimage import exposure
import cv2
import os

# Caminho para as imagens
diretorio_imagens = './data/images/'

# Pré-processamento das imagens
image_size = (224, 224)  # Tamanho ideal para modelos pré-treinados
batch_size = 32

# Aumento de dados para melhorar o desempenho
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Verificar se os diretórios existem, senão usar dataset_passaros
import os

train_dir = './data/train' if os.path.exists('./data/train') else './dataset_passaros/images/train'
val_dir = './data/validation' if os.path.exists('./data/validation') else './dataset_passaros/images/val'

print(f"Usando diretório de treino: {train_dir}")
print(f"Usando diretório de validação: {val_dir}")

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    val_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Construção do modelo
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True

# Descongela apenas as últimas camadas
for layer in base_model.layers[:-30]:  # Ajuste o número se necessário
    layer.trainable = False

# Adicionando camadas de dropout e normalização
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinamento do modelo
checkpoint_cb = tf.keras.callbacks.ModelCheckpoint('melhor_modelo.h5', save_best_only=True)
earlystop_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Pega as classes do generator
labels = train_generator.classes
class_weights = class_weight.compute_class_weight(class_weight='balanced',
                                                  classes=np.unique(labels),
                                                  y=labels)
class_weights_dict = dict(enumerate(class_weights))

# Treinamento do modelo
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size, 
    callbacks=[checkpoint_cb, earlystop_cb], 
    class_weight=class_weights_dict
)

# Salvar o modelo treinado
model.save('modelo_classificacao_passaros.h5')
model.save('data/models/modelo_classificacao_passaros.keras')
print("Treinamento finalizado. Modelo salvo como .h5 e .keras.")

# Extração de características e treinamento do classificador
def extrair_caracteristicas(imagem):
    # Redimensionar a imagem para o tamanho esperado
    imagem = cv2.resize(imagem, (224, 224))
    # Converter para escala de cinza
    imagem_gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
    # Extrair características HOG
    features, hog_image = hog(imagem_gray, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True)
    # Melhorar a visualização do HOG
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return features

# Coletar características e rótulos para o classificador
X = []
y = []

# Iterar sobre o diretório de treinamento
for class_id in os.listdir('./data/train'):
    class_dir = os.path.join('./data/train', class_id)
    for img_file in os.listdir(class_dir):
        img_path = os.path.join(class_dir, img_file)
        imagem = cv2.imread(img_path)
        if imagem is not None:
            features = extrair_caracteristicas(imagem)
            X.append(features)
            y.append(class_id)

# Converter para arrays numpy
X = np.array(X)
y = np.array(y)

# Treinamento do classificador Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X, y)

# Alternativamente, para SVM
# svm_classifier = SVC(kernel='linear')
# svm_classifier.fit(X, y)

print("Classificador treinado com sucesso!")