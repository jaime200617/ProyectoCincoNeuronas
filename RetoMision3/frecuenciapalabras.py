import nltk
from nltk.tokenize import word_tokenize #Divide el texto en palabras
from nltk.corpus import stopwords #Lista de palabras comunes
from nltk.probability import FreqDist #Frecuencia de palabras
from collections import Counter #Frecuencia de palabras

texto = """Puedo escribir los versos más tristes esta noche. Escribir, por ejemplo: La noche está estrellada, 
y tiritan, azules, los astros, a lo lejos. El viento de la noche gira en el cielo y canta. 
Puedo escribir los versos más tristes esta noche. Yo la quise, y a veces ella también me quiso"""

# Tokenizar el texto
palabras = word_tokenize(texto, language="spanish")
#print(palabras)

# Stopwords = eliminar los conectores
stopwords = set(stopwords.words("spanish"))

# Filtrar las palabras que estan en la lista de stopwords
palabras_filtradas = [palabra for palabra in palabras if palabra not in stopwords]
print(palabras_filtradas)

# Frecuencia de palabras
frecuencia = FreqDist(palabras_filtradas)
print(frecuencia.most_common(4))