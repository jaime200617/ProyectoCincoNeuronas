# Ejemplo de clasificación con KNN
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Cargar el conjuno de datos IRIS
iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el clasificador de vecinos mas cercanos
knn = KNeighborsClassifier(n_neighbors=3)

# Entrenar el clasificador
knn.fit(X_train, y_train)

# Predecir las etiquetas para los datos de prueba
y_pred = knn.predict(X_test)

# Calcular la precisión del clasificador
accuracy = accuracy_score(y_test, y_pred)

print('Precisión del clasificador:', accuracy)

