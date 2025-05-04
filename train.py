import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.utils import class_weight

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

train_generator = datagen.flow_from_directory(
    './data/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = datagen.flow_from_directory(
    './data/validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical'
)

base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = True
# Descongela apenas as últimas camadas
for layer in base_model.layers[:-30]:  # Ajuste o número se necessário
    layer.trainable = False

num_classes = len(train_generator.class_indices)
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
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
model.save('modelo_classificacao_passaros.keras')
print("Treinamento finalizado. Modelo salvo como .h5 e .keras.")
