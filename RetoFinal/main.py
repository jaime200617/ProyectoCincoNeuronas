from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Datos de ejemplo
data = [
    {"INGMEN": 1000000, "ESTSOC": 1, "COSMEN": 50000, "PRIORIDAD": 1},
    {"INGMEN": 1500000, "ESTSOC": 2, "COSMEN": 60000, "PRIORIDAD": 1},
    {"INGMEN": 2000000, "ESTSOC": 3, "COSMEN": 70000, "PRIORIDAD": 1},
    {"INGMEN": 1800000, "ESTSOC": 2, "COSMEN": 55000, "PRIORIDAD": 1},
    {"INGMEN": 1200000, "ESTSOC": 1, "COSMEN": 65000, "PRIORIDAD": 1},
    {"INGMEN": 2500000, "ESTSOC": 3, "COSMEN": 80000, "PRIORIDAD": 1},
    {"INGMEN": 2800000, "ESTSOC": 1, "COSMEN": 50000, "PRIORIDAD": 1},
    {"INGMEN": 2200000, "ESTSOC": 2, "COSMEN": 60000, "PRIORIDAD": 1},
    {"INGMEN": 1500000, "ESTSOC": 3, "COSMEN": 90000, "PRIORIDAD": 1},
    {"INGMEN": 2000000, "ESTSOC": 1, "COSMEN": 75000, "PRIORIDAD": 1},
    {"INGMEN": 3500000, "ESTSOC": 1, "COSMEN": 50000, "PRIORIDAD": 0},
    {"INGMEN": 4000000, "ESTSOC": 2, "COSMEN": 60000, "PRIORIDAD": 0},
    {"INGMEN": 4500000, "ESTSOC": 3, "COSMEN": 45000, "PRIORIDAD": 0},
    {"INGMEN": 2000000, "ESTSOC": 4, "COSMEN": 50000, "PRIORIDAD": 0},
    {"INGMEN": 2500000, "ESTSOC": 5, "COSMEN": 60000, "PRIORIDAD": 0},
    {"INGMEN": 3000000, "ESTSOC": 6, "COSMEN": 45000, "PRIORIDAD": 0},
    {"INGMEN": 1000000, "ESTSOC": 1, "COSMEN": 30000, "PRIORIDAD": 0},
    {"INGMEN": 2000000, "ESTSOC": 2, "COSMEN": 20000, "PRIORIDAD": 0},
    {"INGMEN": 1500000, "ESTSOC": 3, "COSMEN": 35000, "PRIORIDAD": 0},
    {"INGMEN": 3200000, "ESTSOC": 1, "COSMEN": 55000, "PRIORIDAD": 0},
]

# Datos de ejemplo para energía limpia
data_energia = [
    {"RADSOL": 6.5, "VELVIE": 12.0, "ENERGIA": "Solar"},
    {"RADSOL": 5.0, "VELVIE": 15.0, "ENERGIA": "Eólica"},
    {"RADSOL": 7.0, "VELVIE": 8.0, "ENERGIA": "Solar"},
    {"RADSOL": 3.5, "VELVIE": 20.0, "ENERGIA": "Eólica"}
]

# Separar características y etiquetas
X = [[item["INGMEN"], item["ESTSOC"], item["COSMEN"]] for item in data]
y = [item["PRIORIDAD"] for item in data]

# Dividir datos y entrenar RandomForestClassifier para clasificación de prioridad
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)

# Preparar datos para el modelo de energía limpia
X_energia = [[item["RADSOL"], item["VELVIE"]] for item in data_energia]
y_energia = [1 if item["ENERGIA"] == "Solar" else 0 for item in data_energia]  # 1: Solar, 0: Eólica

# Dividir datos y entrenar RandomForestClassifier para energía limpia
X_train_ene, X_test_ene, y_train_ene, y_test_ene = train_test_split(X_energia, y_energia, test_size=0.2, random_state=42)
rf_energia = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_energia.fit(X_train_ene, y_train_ene)

# Evaluar el modelo
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión Priorización:", accuracy)
# Evaluar el modelo
y_pred_energia = rf_energia.predict(X_test_ene)
accuracy = accuracy_score(y_test_ene, y_pred_energia)
print("Precisión Energía:", accuracy)

# FastAPI
app = FastAPI()
app.title = "Identificación de Hogares Vulnerables para Energías Limpias"
app.version = "1.0.0"

app.mount("/images", StaticFiles(directory="images"), name="images")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", tags=["Home"])
def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Simulador de Priorización de Energías Limpias</title>
        <link rel="stylesheet" type="text/css" href="/static/styles.css">
    </head>
    <body>
        <h1>Simulador de Energías Limpias</h1>
        <!-- Primer formulario -->
        <form id="form-priorizacion" class="formulario" onsubmit="simulatePriorizacion(event)">
            <h3>Formulario de Priorización</h3>
            <label for="ingresos">Ingresos Mensuales:</label>
            <input 
                type="text" 
                id="ingresos" 
                placeholder="Ingresos Mensuales" 
                required 
                oninput="formatCurrency(this)" 
            />

            <label for="estrato">Estrato Socioeconómico:</label>
            <input type="number" id="estrato" placeholder="Estrato Socioeconómico" required />

            <label for="costo">Costo Energía Mensual:</label>
            <input 
                type="text" 
                id="costo" 
                placeholder="Costo Energía Mensual" 
                oninput="formatCurrency(this)" 
            />

            <button type="submit">Simular Priorización</button>
        </form>
        <!-- Resultado de priorización -->
        <div id="priorizacion-result" style="display: none;">
            <h2 id="priorizacion-message"></h2>
        </div>
        <!-- Segundo formulario -->
        <form id="form-energia" class="formulario" onsubmit="simulateEnergia(event)" style="display: none;">
            <h3>Formulario de Energía Limpia</h3>
            <label for="radiacion">Radiación Solar (kWh/m²):</label>
            <input type="number" step="0.1" id="radiacion" placeholder="Radiación Solar (kWh/m²)" required />

            <label for="viento">Velocidad del Viento (m/s):</label>
            <input type="number" step="0.1" id="viento" placeholder="Velocidad del Viento (m/s)" required />
            <button type="submit">Simular Energía Limpia</button>
        </form>
        <h2 id="energia-message" style="margin-top: 20px;"></h2>

        <script>
            function formatCurrency(input) {
                // Eliminar caracteres no numéricos
                let value = input.value.replace(/\D/g, "");
                
                // Formatear como moneda
                value = new Intl.NumberFormat("es-CO", {
                    style: "currency",
                    currency: "COP",
                    minimumFractionDigits: 0
                }).format(value);
                
                // Actualizar el valor del campo de entrada
                input.value = value;
            }

            async function simulatePriorizacion(event) {
                event.preventDefault();

                // Obtener y procesar el valor del campo "ingresos"
                const ingresosField = document.getElementById("ingresos");
                const ingresos = ingresosField.value.replace(/\D/g, ""); // Eliminar formato de moneda para obtener solo el número

                const estrato = document.getElementById("estrato").value;

                const costoField = document.getElementById("costo");
                const costo = costoField.value.replace(/\D/g, ""); // Eliminar formato de moneda para obtener solo el número

                // Enviar la solicitud al servidor
                const response = await fetch(`/simulador?Ingresos=${ingresos}&Estrato=${estrato}&Costo=${costo}`);
                const result = await response.json();

                // Actualizar los mensajes y mostrar el segundo formulario si aplica
                const message = document.getElementById("priorizacion-message");
                const resultDiv = document.getElementById("priorizacion-result");
                const energiaForm = document.getElementById("form-energia");

                // Reiniciar estados previos
                resultDiv.style.display = "block";
                message.textContent = result["Predicción de Priorización"];
                message.className = ""; // Limpia clases previas

                // Asignar clases según el resultado
                if (result["Predicción de Priorización"] === "Sí Priorizado") {
                    message.classList.add("si");
                    energiaForm.style.display = "block"; // Mostrar el segundo formulario
                } else {
                    message.classList.add("no");
                }
            }

            async function simulateEnergia(event) {
                event.preventDefault();

                // Obtener los valores del formulario
                const radiacion = document.getElementById("radiacion").value;
                const viento = document.getElementById("viento").value;

                // Enviar la solicitud al servidor
                const response = await fetch(`/energia?Radiacion=${radiacion}&Viento=${viento}`);
                const result = await response.json();

                // Mostrar el mensaje de energía recomendada
                const energiaMessage = document.getElementById("energia-message");
                energiaMessage.textContent = `Energía Recomendada: ${result["Energía Recomendada"]}`;
                energiaMessage.className = ""; // Limpia clases previas

                // Asignar clases según el tipo de energía
                if (result["Energía Recomendada"] === "Solar") {
                    energiaMessage.classList.add("solar");
                } else if (result["Energía Recomendada"] === "Eólica") {
                    energiaMessage.classList.add("eolica");
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/simulador", tags=["Simulador"])
def get_simulador(Ingresos: int = Query(...), Estrato: int = Query(...), Costo: int = Query(...)):
    new_data = [[Ingresos, Estrato, Costo]]
    prediction = rf.predict(new_data)
    prediction_label = "Sí Priorizado" if prediction[0] == 1 else "No Priorizado"
    return JSONResponse(content={"Predicción de Priorización": prediction_label})

@app.get("/energia", tags=["Energía Limpia"])
def get_energia(Radiacion: float = Query(...), Viento: float = Query(...)):
    new_data = [[Radiacion, Viento]]
    prediction = rf_energia.predict(new_data)
    prediction_label = "Solar" if prediction[0] == 1 else "Eólica"
    return JSONResponse(content={"Energía Recomendada": prediction_label})