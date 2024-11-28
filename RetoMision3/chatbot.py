from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

nltk.download('punkt')

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

data = [
    ("super deliciosa", "positive"),
    ("muy costosa", "negative"),
    ("q buena opcion", "positive"),
    ("muy fria", "negative"),
    ("estaba fria", "negative"),
    ("es la mejor del mundo", "positive"),
    ("no me guto la salsa", "negative"),
    ("es una explision de sabor", "positive"),
    ("se demoro mucho", "negative"),
    ("cada vez q voy me enamoro mas", "positive"),
    ("me encanta el pan" , "positive"),
    ("The special effects in this movie were impressive", "positive"),
    ("ya le dije a todos que no me gusta", "negative"),
    ("muy caliente", "positive"),
    ("esto es una maravilla" , "positive"),
    ("recomendadísimo" , "positive"),
    ("qué buenas salsas" , "positive"),
    ("volveré a comer en este sitio" , "positive"),
    ("la mejor hamburguesa que he comido en mi vida" , "positive"),
    ("gracias por esta experiencia " , "positive"),
    ("la mejor hamburguesa del mundo" , "positive"),
    ("es mi hamburguesa favorita" , "positive"),
    ("que asco de hamburguesa" , "negative"),
    ("una porqueria" , "negative"),
    ("la peor hamburguesa que he probado" , "negative"),
    ("les falta magia" , "negative"),
    ("no la recomiendo" , "negative"),
    ("no volveré a comer de esta hamburguesa" , "negative"),
    ("no satisface mis expectativas" , "negative"),
    ("mal cocinada" , "negative"),
    ("mal servida" , "negative"),
    ("no me gustó nada" , "negative"),
    ("Cual es el número de teléfono de contacto?", "telefono"),
    ("Cual es el correo de contacto?", "correo"),
    ("Cual es el instagram de contacto?", "instagram"),
    ("Cules son las redes sociales", "instagram"),
    ("Cuales son los horarios de atencion?", "horarios"),
    ("En que horario atienden", "horarios"),
    ("A que horas abren", "horarios"),
    ("Que horas manejan", "horarios"),
    ("A que horas cierran", "horarios"),
    ("Abren los domingos?", "horarios"),
    ("no me gusto el producto como lo devuelvo" , "devolucion"),
    ("no me siento nada satisfecho con el producto deseo devolverlo" , "devolucion"),
    ("necesito que me reciban de vuelta el producto" , "devolucion"),
    ("como me comunico con el gerente para un tema de devolución" , "devolucion"),
    ("cuánto vale la hamburguesa clásica", "precios"),
    ("cuál es el precio de la hamburguesa clásica", "precios"),
    ("qué precio tiene la hamburguesa clásica", "precios"),
    ("me podrías decir el valor de la hamburguesa clásica", "precios"),
    # Hamburguesa Monster
    ("cuánto cuesta la hamburguesa Monster", "precios"),
    ("cuál es el valor de la hamburguesa Monster", "precios"),
    ("qué precio tiene la hamburguesa Monster", "precios"),
    ("quiero saber el precio de la hamburguesa Monster", "precios"),
    # Hamburguesa Súper
    ("cuánto vale la hamburguesa súper", "precios"),
    ("cuál es el precio de la hamburguesa súper", "precios"),
    ("qué precio tiene la hamburguesa súper", "precios"),
    ("me puedes decir cuánto cuesta la hamburguesa súper", "precios"),
    # Hamburguesa BBQ Lover
    ("cuánto cuesta la hamburguesa BBQ Lover", "precios"),
    ("cuál es el precio de la hamburguesa BBQ Lover", "precios"),
    ("quiero saber cuánto vale la hamburguesa BBQ Lover", "precios"),
    ("qué precio tiene la hamburguesa BBQ Lover", "precios"),
    # Hamburguesa Veggie Power
    ("cuánto vale la hamburguesa Veggie Power", "precios"),
    ("cuál es el valor de la hamburguesa Veggie Power", "precios"),
    ("quiero saber el precio de la hamburguesa Veggie Power", "precios"),
    ("qué precio tiene la hamburguesa Veggie Power", "precios"),
    # Hamburguesa Picante Extrema
    ("cuánto cuesta la hamburguesa Picante Extrema", "precios"),
    ("me puedes decir el precio de la hamburguesa Picante Extrema", "precios"),
    ("qué precio tiene la hamburguesa Picante Extrema", "precios"),
    ("cuál es el valor de la hamburguesa Picante Extrema", "precios"),
    # Hamburguesa Doble Queso
    ("quiero saber cuánto vale la hamburguesa Doble Queso", "precios"),
    ("cuál es el precio de la hamburguesa Doble Queso", "precios"),
    ("qué precio tiene la hamburguesa Doble Queso", "precios"),
    ("cuánto cuesta la hamburguesa Doble Queso", "precios"),
    # Hamburguesa Gourmet Deluxe
    ("cuánto vale la hamburguesa Gourmet Deluxe", "precios"),
    ("cuál es el precio de la hamburguesa Gourmet Deluxe", "precios"),
    ("quiero saber cuánto cuesta la hamburguesa Gourmet Deluxe", "precios"),
    ("qué precio tiene la hamburguesa Gourmet Deluxe", "precios"),
    # Hamburguesa Crispy Bacon
    ("cuánto cuesta la hamburguesa Crispy Bacon", "precios"),
    ("cuál es el precio de la hamburguesa Crispy Bacon", "precios"),
    ("quiero saber cuánto vale la hamburguesa Crispy Bacon", "precios"),
    ("qué precio tiene la hamburguesa Crispy Bacon", "precios"),
    ("me puedes decir el valor de la hamburguesa Crispy Bacon", "precios"),
    ("Existen los unicornios", "unknown"),
    ("Existen bailes de dragones", "unknown"),
    ("Existen los extraterrestres", "unknown"),
    ("La hamburguesa estaba caliente y deliciosa", "positive"),
    ("Me encanta lo caliente y jugosa que estaba", "positive"),
    ("La comida llegó bien caliente, excelente", "positive"),
    ("Siempre sirven todo caliente y recién hecho", "positive"),
    ("Las papas estaban calientes y crujientes, perfectas", "positive"),
    ("Nada mejor que una hamburguesa bien caliente", "positive"),
    ("La bebida estaba a la temperatura perfecta, muy caliente", "positive"),
    ("Todo estuvo caliente, fresco y delicioso", "positive"),
    ("Disfruté mucho porque estaba caliente y sabrosa", "positive"),
    ("La sopa estuvo muy caliente y reconfortante", "positive"),
    ("La hamburguesa estaba fría, qué decepción", "negative"),
    ("Las papas llegaron completamente frías", "negative"),
    ("La comida fría arruinó mi experiencia", "negative"),
    ("No me gustó que todo estuviera frío", "negative"),
    ("La carne estaba fría y sin sabor", "negative"),
    ("El pan estaba frío y duro", "negative"),
    ("Todo llegó frío, muy mala experiencia", "negative"),
    ("La bebida estaba caliente en lugar de fría", "negative"),
    ("La comida fría es lo peor", "negative"),
    ("Las alitas estaban frías, no me gustaron", "negative"),
    ("¿A qué hora abren?", "horarios"),
    ("¿A qué hora cierran?", "horarios"),
    ("¿Cuáles son sus horarios de apertura?", "horarios"),
    ("¿Cuándo están abiertos?", "horarios"),
    ("¿Puedo ir en la mañana?", "horarios"),
    ("¿Están abiertos por la noche?", "horarios"),
    ("¿Trabajan los fines de semana?", "horarios"),
    ("¿Qué días están abiertos?", "horarios"),
    ("¿Puedo visitarlos un domingo?", "horarios"),
    ("¿Qué horarios manejan para los sábados?", "horarios"),
    ("¿A qué hora comienzan a atender?", "horarios"),
    ("¿Puedo pasar después de las 10 PM?", "horarios"),
    ("¿Cierran a las 9 PM?", "horarios"),
    ("¿Están disponibles los lunes por la tarde?", "horarios"),
    ("¿Qué horario tienen los días festivos?", "horarios"),
    ("¿Abren en navidad?", "horarios"),
    ("¿Hay horarios especiales en año nuevo?", "horarios"),
    ("¿Hasta qué hora trabajan entre semana?", "horarios"),
    ("¿Están abiertos a la hora del almuerzo?", "horarios"),
    ("¿Puedo pedir desayuno temprano?", "horarios"),
    ("¿Atienden en las mañanas?", "horarios"),
    ("¿Están abiertos en la tarde?", "horarios"),
    ("¿Hasta qué hora puedo ir en la noche?", "horarios"),
    ("¿Tienen horario corrido?", "horarios"),
    ("¿A qué hora cierran el sábado?", "horarios"),
    ("¿Abren los domingos por la mañana?", "horarios"),
    ("¿Tienen horario extendido?", "horarios"),
    ("¿Los domingos trabajan hasta tarde?", "horarios"),
    ("¿A qué hora puedo ir el viernes por la noche?", "horarios"),
    ("¿Qué horario tienen el lunes por la tarde?", "horarios"),
    ("¿Cuál es su correo?", "correo"),
    ("¿Me pueden proporcionar un correo electrónico?", "correo"),
    ("¿Cómo puedo enviarles un correo?", "correo"),
    ("¿Tienen email de contacto?", "correo"),
    ("Necesito un correo para escribirles", "correo"),
    ("¿Cuál es el email para comunicarme con ustedes?", "correo"),
    ("¿Puedo obtener su correo electrónico?", "correo"),
    ("Dame su correo electrónico, por favor", "correo"),
    ("¿Dónde puedo enviar un email?", "correo"),
    ("¿Cuentan con un correo para soporte?", "correo"),
    ("¿Puedo mandarles un email?", "correo"),
    ("¿Cuál es su dirección de correo electrónico?", "correo"),
    ("Quiero enviar un correo, ¿a qué dirección?", "correo"),
    ("¿Qué email tienen para atención al cliente?", "correo"),
    ("¿Cuál es su número de teléfono?", "telefono"),
    ("¿Me pueden dar un teléfono de contacto?", "telefono"),
    ("¿Cómo puedo comunicarme por teléfono?", "telefono"),
    ("¿Tienen un número para atención?", "telefono"),
    ("¿Me puedes dar un teléfono para llamarlos?", "telefono"),
    ("Necesito el número de contacto", "telefono"),
    ("¿Tienen un teléfono para soporte?", "telefono"),
    ("¿Puedo llamar a algún número?", "telefono"),
    ("Dame el número de contacto, por favor", "telefono"),
    ("¿Cuál es el número para servicio al cliente?", "telefono"),
    ("¿Qué número puedo usar para comunicarme?", "telefono"),
    ("¿Tienen una línea telefónica?", "telefono"),
    ("¿Dónde los puedo llamar?", "telefono"),
    ("¿Cuál es su número de atención al cliente?", "telefono"),
    ("¿Tienen redes sociales?", "instagram"),
    ("¿Cuál es su cuenta de Instagram?", "instagram"),
    ("¿Me pueden dar su Instagram?", "instagram"),
    ("¿Cómo los encuentro en Instagram?", "instagram"),
    ("¿Tienen Facebook o Instagram?", "instagram"),
    ("¿Puedo seguirlos en redes sociales?", "instagram"),
    ("¿Están en Instagram?", "instagram"),
    ("¿Qué redes sociales manejan?", "instagram"),
    ("¿Cuentan con algún perfil en redes sociales?", "instagram"),
    ("¿Me pueden pasar sus redes sociales?", "instagram"),
    ("Dame su Instagram, por favor", "instagram"),
    ("¿Dónde los encuentro en redes sociales?", "instagram"),
    ("¿Cómo los busco en Instagram?", "instagram"),
    ("¿Tienen un perfil de Instagram?", "instagram"),
    ("¿Cómo hago para devolver un producto?", "devolucion"),
    ("¿Qué debo hacer para iniciar una devolución?", "devolucion"),
    ("¿Aceptan devoluciones?", "devolucion"),
    ("Quiero devolver un producto, ¿cómo lo hago?", "devolucion"),
    ("¿Tienen política de devoluciones?", "devolucion"),
    ("¿Cómo puedo devolver una hamburguesa?", "devolucion"),
    ("Quiero un reembolso, ¿cómo lo gestiono?", "devolucion"),
    ("¿Puedo devolver algo si no me gustó?", "devolucion"),
    ("¿Qué necesito para devolver un producto?", "devolucion"),
    ("¿Cómo gestiono una devolución?", "devolucion"),
    ("¿Qué requisitos tienen para aceptar devoluciones?", "devolucion"),
    ("No estoy satisfecho, quiero devolver el producto", "devolucion"),
    ("No me gustó el producto, ¿puedo devolverlo?", "devolucion"),
    ("La hamburguesa llegó mal, quiero un reembolso", "devolucion"),
    ("El producto no era lo que esperaba, quiero devolverlo", "devolucion"),
    ("Quiero que me reembolsen el dinero", "devolucion"),
    ("El pedido llegó mal, necesito devolverlo", "devolucion"),
    ("El producto no es lo que pedí, ¿cómo lo devuelvo?", "devolucion"),
    ("La comida estaba mal preparada, quiero una devolución", "devolucion"),
    ("Quiero que recojan el producto y me devuelvan el dinero", "devolucion"),
    ("Me llegó algo diferente, quiero devolverlo", "devolucion"),
    ("¿Qué pasos debo seguir para devolver un producto?", "devolucion"),
    ("¿Dónde debo llevar el producto para devolverlo?", "devolucion"),
    ("¿Debo llamar para solicitar una devolución?", "devolucion"),
    ("¿Cuánto tiempo tengo para devolver algo?", "devolucion"),
    ("¿Cuánto tiempo tarda el reembolso?", "devolucion"),
    ("¿Cómo obtengo mi dinero de vuelta?", "devolucion"),
    ("¿Puedo devolver un producto parcialmente usado?", "devolucion"),
    ("¿Aceptan devoluciones por pedidos a domicilio?", "devolucion"),
    ("¿Puedo devolver un pedido si llegó tarde?", "devolucion"),
    ("¿Tienen una política para devoluciones de productos dañados?", "devolucion")
]

respuestas = {
    "positive": ["¡Me alegra que te guste!", "¡Te esperamos de nuevo!"],
    "negative": ["Lo sentimos. Vamos a trabajar en esto.", "Vamos a mejorar, gracias por tu retroalimentación."],
    "telefono": ["Nuestro número de teléfono es 018000123.", "Puedes llamarnos al 018000123."],
    "correo": ["Nuestro correo es HCincoNeuronas@gmail.com.", "Nos puedes escribir al correo HCincoNeuronas@gmail.com."],
    "instagram": ["Lo siento, por ahora no tenemos Instagram.", "No contamos con redes sociales por ahora."],
    "horarios": ["Los horarios son de martes a domingo de 12:00 a 22:00.", "Atendemos de martes a domingo de 12:00 a 22:00."],
    "devolucion": [
        "Ya lo comunicamos con el gerente.",
        "Por favor, comuníquese con la línea de servicio al cliente.",
        "Déjenos su número y nos comunicaremos con usted."
    ],
    "precios": [
        "Los precios están disponibles en la carta.",
        "Puedes ver los precios en nuestro menú.",
        "Por favor, consulta la carta para conocer los precios."
    ],
    "unknown": ["No entendí tu pregunta. ¿Puedes intentar de nuevo o ser más específico?"]
}
       
# Preprocesamiento de datos: Tokenización y extracción de características

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return {word: True for word in tokens}

# Aplicamos el preprocesamiento a los datos
featuresets = [(preprocess(text), sentimiento) for text, sentimiento in data]

# Entrenamos un clasificador utilizando Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(featuresets)

# Chatbot
def chatbot(frase_usuario):
    categoria = preprocess(frase_usuario)

    # Obtener la clasficación y probabilidad de la frase
    probabilidades = classifier.prob_classify(categoria)
    prediccion = probabilidades.max()
    probabilidad = probabilidades.prob(prediccion)  

    # Definir umbral de probabilidad minima
    umbral = 0.3
    
    if probabilidad >= umbral and prediccion in respuestas:
        return random.choice(respuestas[prediccion])
    else:
        return "No entendí tu pregunta. ¿Puedes intentar de nuevo o ser más específico?"

@app.get("/chatbot2", tags=["Chatbot"])
def get_respuesta(Conversacion: str = Query(...)):
    prediction = chatbot(Conversacion)
    return JSONResponse(content={"Respuesta a tu conversación": prediction})

@app.get("/", tags=["Home"])
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Hamburguesas Cinco Neuronas</title>
        <link rel="stylesheet" type="text/css" href="/static/stylesheet.css">
    </head>
    <body>
        <h1>Hamburguesas Cinco Neuronas</h1>
        <!-- Primer formulario -->
        <form id="chatbot-form" class="formulario" onsubmit="simularRespuesta(event)">
            <h3>ChatBot</h3>
            <label for="conversacion">Ingresa tu comentario por favor:</label>
            <input 
                type="text" 
                id="conversacion" 
                placeholder="Conversación" 
                required                  
            />

            <button type="submit">Enviar</button>
        </form>
        <!-- Resultado de conversación -->
        <div id="resultado">
            <h2 id="resultado_message">¡Escribe un comentario para empezar!</h2>
        </div>

        <script>
            async function simularRespuesta(event) {
                event.preventDefault();

                // Obtener y procesar el valor del campo conversación
                const conversacion = document.getElementById("conversacion");
               
                // Enviar la solicitud al servidor
                const response = await fetch(`/chatbot2?Conversacion=${conversacion.value}`);
                const result = await response.json();

                // Actualizar los mensajes y mostrar el segundo formulario si aplica
                const message = document.getElementById("resultado_message");
                const resultDiv = document.getElementById("resultado");

                message.textContent = result["Respuesta a tu conversación"];
                resultDiv.style.display = "block";
                
                // Limpiar el campo conversación
                document.getElementById("conversacion").value = "";
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

