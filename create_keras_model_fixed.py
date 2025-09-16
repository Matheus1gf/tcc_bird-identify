#!/usr/bin/env python3
"""
Script para criar um modelo Keras funcional para classificação de pássaros
"""

import tensorflow as tf
import numpy as np
import os

def create_simple_keras_model():
    """Cria um modelo Keras simples e funcional"""
    
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
    
    # Compilar com otimizador simples
    model.compile(
        optimizer='adam',  # Usar string simples
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Função principal"""
    print("🔧 Criando modelo Keras funcional...")
    
    try:
        # Criar modelo
        model = create_simple_keras_model()
        
        # Criar dados dummy para treinar o modelo
        print("📊 Criando dados de treinamento dummy...")
        x_train = np.random.random((100, 224, 224, 3)).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 100), 10)
        
        # Treinar por algumas épocas
        print("🚀 Treinando modelo...")
        model.fit(x_train, y_train, epochs=1, verbose=1)
        
        # Salvar modelo
        model_path = "modelo_classificacao_passaros.keras"
        print(f"💾 Salvando modelo em: {model_path}")
        model.save(model_path)
        
        print("✅ Modelo Keras criado com sucesso!")
        print(f"📁 Arquivo salvo: {os.path.abspath(model_path)}")
        
        # Testar carregamento
        print("🧪 Testando carregamento do modelo...")
        loaded_model = tf.keras.models.load_model(model_path)
        print("✅ Modelo carregado com sucesso!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro ao criar modelo: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Modelo Keras funcional criado!")
    else:
        print("\n💥 Falha ao criar modelo Keras")
