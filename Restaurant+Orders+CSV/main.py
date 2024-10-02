import tensorflow as tf
import numpy as np

# Datos de entrada y salida
x_train = np.array([1.0, 2.0, 3.0, 4.0])
y_train = np.array([2.0, 4.0, 6.0, 8.0])

# Modelo de regresión lineal con TensorFlow/Keras
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilar modelo
modelo.compile(optimizer='sgd', loss='mean_squared_error')

# Entrenar el modelo
modelo.fit(x_train, y_train, epochs=10)

# Predecir (convertir lista a NumPy array)
predicciones = modelo.predict(np.array([5.0]))
print("Predicción para 5.0:", predicciones)
