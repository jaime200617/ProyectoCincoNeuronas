import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  

sentences = "Esta es una prueba de tokenizacion de texto"
tokens = word_tokenize(sentences)
print(tokens)   


