#!/usr/bin/env python3
"""
Criar modelo Keras simples e funcional para classificação de pássaros
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_keras_model():
    """Cria um modelo Keras simples para classificação de pássaros"""
    
    print("🔧 Criando modelo Keras simples...")
    
    # Criar modelo sequencial simples
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')  # 10 classes de pássaros
    ])
    
    # Compilar modelo com otimizador compatível
    model.compile(
        optimizer='adam',  # Usar string em vez de objeto Adam
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Criar dados dummy para treinar o modelo
    print("📊 Criando dados dummy para treinamento...")
    dummy_x = np.random.random((100, 224, 224, 3)).astype(np.float32)
    dummy_y = tf.keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)
    
    # Treinar modelo com dados dummy
    print("🚀 Treinando modelo...")
    model.fit(dummy_x, dummy_y, epochs=1, batch_size=32, verbose=1)
    
    # Salvar modelo
    model_path = "modelo_classificacao_passaros.keras"
    print(f"💾 Salvando modelo em: {model_path}")
    model.save(model_path)
    
    print("✅ Modelo Keras criado com sucesso!")
    return model_path

if __name__ == "__main__":
    try:
        create_simple_keras_model()
    except Exception as e:
        print(f"❌ Erro ao criar modelo: {e}")
        import traceback
        traceback.print_exc()
