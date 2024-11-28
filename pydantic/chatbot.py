from fastapi import FastAPI
from pydantic import BaseModel
import random

app = FastAPI()

# Diccionario de categorías con palabras clave y respuestas
categorias = {
    "saludo": {
        "palabras_claves": ["hola", "buenos dias", "buenas tardes", "buenas noches"],
        "respuestas": ["Hola, ¿qué tal?", "Buenos días, un gusto saludarte", "Buenas tardes, un gusto saludarte", "Buenas noches, que descanses"]
    },
    "despedida": {
        "palabras_claves": ["adios", "chao", "hasta luego", "nos vemos", "bye"],
        "respuestas": ["Gracias por su visita", "Un gusto atenderlo", "Que tenga un buen día", "Nos vemos pronto"]
    },
    "precio": {
        "palabras_claves": ["precio", "cuánto cuesta", "cuánto vale", "cuánto es", "valor"],
        "respuestas": ["El precio depende del modelo del celular. ¿Cuál te interesa?"]
    },
    "modelo1": {
        "palabras_claves": ["basico", "intermedio", "avanzado"],
        "respuestas": ["El precio del modelo uno es de 300.000 pesos en básico, 400.000 pesos en intermedio y 500.000 pesos en avanzado. ¿Cuál te interesa?"]
    },
    "modelo2": {
        "palabras_claves": ["xl plus", "xl lite", "xl pro"],
        "respuestas": ["El precio del modelo XL Plus es de 400.000 pesos en básico, 500.000 pesos el XL Lite y 600.000 pesos el XL Pro. ¿Cuál te interesa?"]
    }
}

# Clasificador de categorías
def clasificar_categoria(frase):
    frase = frase.lower()
    for categoria, data in categorias.items():
        if any(palabra_clave in frase for palabra_clave in data["palabras_claves"]):
            return categoria
    return "desconocido"

# Chatbot
def chatbot(frase_usuario):
    categoria = clasificar_categoria(frase_usuario)
    if categoria == "desconocido":
        return "Lo siento, no entendí tu pregunta. Por favor, sea más específico."
    return random.choice(categorias[categoria]["respuestas"])

# Modelo para entrada de datos
class FraseEntrada(BaseModel):
    frase: str

# Endpoint del chatbot
@app.post("/chatbot/")
def obtener_respuesta(entrada: FraseEntrada):
    respuesta = chatbot(entrada.frase)
    return {"respuesta": respuesta}