from fastapi import FastAPI
from pydantic import BaseModel # Implementacion de la Libreria pydantic

app = FastAPI()

class usuario(BaseModel):
    nombre: str
    apellido: str
    edad: int
    correo: str
    
@app.post("/crear_usuario")
async def crear_usuario(usuario: usuario):
    return {"mensaje": f"Usuario {usuario.nombre} creado con exito"}