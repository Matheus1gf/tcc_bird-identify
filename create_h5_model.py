#!/usr/bin/env python3
"""
Script para criar um modelo Keras em formato HDF5
"""

import tensorflow as tf
import numpy as np
import os

def create_h5_model():
    """Cria um modelo Keras em formato HDF5"""
    
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
    
    # Salvar em formato HDF5
    h5_path = "modelo_classificacao_passaros.h5"
    model.save(h5_path)
    
    print(f"✅ Modelo HDF5 criado em: {h5_path}")
    
    # Testar carregamento
    print("🧪 Testando carregamento...")
    loaded_model = tf.keras.models.load_model(h5_path)
    print("✅ Modelo HDF5 carregado com sucesso!")
    
    return h5_path

def main():
    """Função principal"""
    print("🔧 Criando modelo Keras HDF5...")
    
    try:
        h5_path = create_h5_model()
        
        # Copiar para o nome esperado pelo sistema
        keras_path = "modelo_classificacao_passaros.keras"
        os.system(f"cp {h5_path} {keras_path}")
        
        print(f"✅ Modelo copiado para: {keras_path}")
        print("\n🎉 Modelo Keras HDF5 criado com sucesso!")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n💥 Falha ao criar modelo Keras")
