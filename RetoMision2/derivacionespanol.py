import nltk
from nltk.stem import SnowballStemmer

# Definir una lista de palabras en español
palabras = ["corriendo", "jugando", "saltado"]

# Crear una instancia de SnowballStemmer para español
stemmer = SnowballStemmer("spanish")

# Aplicar el stemming a cada palabra de la lista
stems = [stemmer.stem(palabra) for palabra in palabras]

# Imprimir los resultados
print(stems)