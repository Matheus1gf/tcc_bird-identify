#!/usr/bin/env python3
"""
Script para criar apenas os pesos do modelo Keras
"""

import tensorflow as tf
import numpy as np
import os

def create_model_weights_only():
    """Cria apenas os pesos do modelo"""
    
    # Modelo muito simples
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    # Criar dados dummy mínimos
    x_train = np.random.random((5, 224, 224, 3)).astype(np.float32)
    y_train = tf.keras.utils.to_categorical(np.random.randint(0, 10, 5), 10)
    
    # Compilar sem otimizador personalizado
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Treinar por 1 época
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    # Salvar apenas a arquitetura e pesos
    model_path = "modelo_classificacao_passaros.keras"
    
    # Salvar usando save_weights_only
    weights_path = "modelo_weights.h5"
    model.save_weights(weights_path)
    
    # Salvar arquitetura separadamente
    architecture_path = "modelo_architecture.json"
    with open(architecture_path, 'w') as f:
        f.write(model.to_json())
    
    print(f"✅ Pesos salvos em: {weights_path}")
    print(f"✅ Arquitetura salva em: {architecture_path}")
    
    return model, weights_path, architecture_path

def create_keras_file():
    """Cria arquivo .keras funcional"""
    
    try:
        model, weights_path, architecture_path = create_model_weights_only()
        
        # Criar novo modelo a partir da arquitetura
        with open(architecture_path, 'r') as f:
            model_json = f.read()
        
        new_model = tf.keras.models.model_from_json(model_json)
        new_model.load_weights(weights_path)
        
        # Salvar como .keras sem otimizador
        keras_path = "modelo_classificacao_passaros.keras"
        
        # Usar tf.saved_model.save que é mais compatível
        tf.saved_model.save(new_model, keras_path)
        
        print(f"✅ Modelo .keras criado em: {keras_path}")
        
        # Testar carregamento
        print("🧪 Testando carregamento...")
        loaded_model = tf.saved_model.load(keras_path)
        print("✅ Modelo carregado com sucesso!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

def main():
    """Função principal"""
    print("🔧 Criando modelo Keras compatível...")
    
    success = create_keras_file()
    
    if success:
        print("\n🎉 Modelo Keras compatível criado!")
    else:
        print("\n💥 Falha ao criar modelo Keras")

if __name__ == "__main__":
    main()
