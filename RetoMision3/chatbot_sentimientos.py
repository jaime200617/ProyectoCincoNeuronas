from fastapi import FastAPI
from pydantic import BaseModel
import random
import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

# Inicializar NLTK
nltk.download("punkt")

# Crear la aplicación FastAPI
app = FastAPI()

# Dataset de entrenamiento con frases etiquetadas
entrenamiento = [
    ("Me siento muy feliz", "positivo"),
    ("Estoy muy contento", "positivo"),
    ("Esto es fantástico", "positivo"),
    ("Me encanta lo que está pasando", "positivo"),
    ("Estoy molesto", "negativo"),
    ("Esto no me gusta", "negativo"),
    ("Estoy muy triste", "negativo"),
    ("Es terrible lo que ocurrió", "negativo"),
    ("No me siento bien", "negativo")
]

# Diccionario de respuestas según sentimiento
respuestas_sentimientos = {
    "positivo": [
        "Me alegra que estes bien",
        "Me parece genial",
        "Fantástico"
    ],
    "negativo": [
        "Lo lamento mucho",
        "Lo siento",
        "Espero que todo mejore"
    ]
}

# Preparar los datos para el clasificador
def extraer_caracteristicas(frase):
    palabras = word_tokenize(frase.lower())
    return {palabra: True for palabra in palabras}

# Preparar los datos de entrenamiento
caracteristicas_entrenamiento = [(extraer_caracteristicas(frase), sentimiento) for frase, sentimiento in entrenamiento]

# Entrenar el clasificador Naive Bayes
clasificador = NaiveBayesClassifier.train(caracteristicas_entrenamiento)

# Chatbot
def chatbot(frase_usuario):
    caracteristicas = extraer_caracteristicas(frase_usuario)
    sentimiento = clasificador.classify(caracteristicas)
    return random.choice(respuestas_sentimientos[sentimiento])

# Modelo para entrada de datos
class FraseEntrada(BaseModel):
    frase: str

# Endpoint del chatbot
@app.post("/chatbot/")
def obtener_respuesta(entrada: FraseEntrada):
    respuesta = chatbot(entrada.frase)
    return {"respuesta": respuesta}