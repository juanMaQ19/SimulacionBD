import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Cargar el dataset
df = pd.read_csv('menu_items.csv')

# Imprimir las primeras filas de los datos
print(df.head())

# Imprimir dimensiones
print(df.shape)  # dimensiones

# Información general del dataset
print(df.info())  # general

# Resumen estadístico
print(df.describe())  # resumen estadístico

# Comprobar valores nulos
print(df.isnull().sum())  # valores nulos

# Limpiar datos eliminando valores nulos o irrelevantes
df = df.dropna()

# Convertir las variables categóricas en variables dummy
df = pd.get_dummies(df, columns=['category'])

# Escalado de datos
# Elegir columnas a escalar
columnas_a_escalar = ['price']
scaler = StandardScaler()
df[columnas_a_escalar] = scaler.fit_transform(df[columnas_a_escalar])

# Separación de características y etiquetas
x = df.drop(['menu_item_id', 'item_name'], axis=1)  # INPUTS
y = df[['category_American']]  # OUTPUTS (lo que queremos predecir)

# División del dataset en entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Modelo
modelo = DecisionTreeClassifier()

# Entrenar el modelo
modelo.fit(x_train, y_train)

# Predicciones
y_pred = modelo.predict(x_test)

# Evaluando el modelo con la métrica de precisión
precision = accuracy_score(y_test, y_pred)
print('Precisión: ' + str(precision))
