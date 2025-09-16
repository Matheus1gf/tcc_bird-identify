#!/usr/bin/env python3
"""
Script para criar um modelo Keras minimalista e funcional
"""

import tensorflow as tf
import numpy as np
import os

def create_minimal_model():
    """Cria um modelo Keras minimalista"""
    
    # Modelo muito simples
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Compilar sem otimizador personalizado
    model.compile(
        optimizer='sgd',  # SGD é mais simples que Adam
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """Função principal"""
    print("🔧 Criando modelo Keras minimalista...")
    
    try:
        # Criar modelo
        model = create_minimal_model()
        
        # Criar dados dummy mínimos
        print("📊 Criando dados mínimos...")
        x_train = np.random.random((10, 224, 224, 3)).astype(np.float32)
        y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 10), 10)
        
        # Treinar por 1 época
        print("🚀 Treinando modelo...")
        model.fit(x_train, y_train, epochs=1, verbose=0)
        
        # Salvar modelo
        model_path = "modelo_classificacao_passaros.keras"
        print(f"💾 Salvando modelo em: {model_path}")
        
        # Salvar sem otimizador personalizado
        model.save(model_path, save_format='keras')
        
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
        print("\n🎉 Modelo Keras minimalista criado!")
    else:
        print("\n💥 Falha ao criar modelo Keras")
