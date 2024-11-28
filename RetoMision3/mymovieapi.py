from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize

# Descarga 'punkt' para tokenización
nltk.download('punkt')

# Ruta al archivo CSV
path = "DataSet/netflix_titles.csv"

# Carga el dataset
def load_movies():
    df = pd.read_csv(path)
    # Selección de columnas del dataset y renombramiento
    movies = df[['show_id', 'title', 'release_year', 'listed_in', 'rating','description']].rename(
        columns={
            'show_id': 'id',
            'title': 'title',
            'release_year': 'year',
            'listed_in': 'category',
            'rating': 'rating',
            'description': 'overview'
        }
    )

    # Reemplaza NaN por valores predeterminados para evitar errores de JSON
    movies = movies.fillna({
        'id': '',
        'title': '',
        'year': 0,
        'category': '',
        'rating': '',
        'description': ''
    })
    
    # Convierte el DataFrame a una lista de diccionarios
    return movies.to_dict(orient="records")

# Carga inicial de las películas
movies_list = load_movies()

# Clasificación de categorías
data = [
    ("Action & Adventure", "ME GUSTA"),
    ("Anime Features", "NO ME GUSTA"),
    ("Anime Series", "ME GUSTA"),
    ("British TV Shows", "NO ME GUSTA"),
    ("Children & Family Movies", "ME GUSTA"),
    ("Classic & Cult TV", "NO ME GUSTA"),
    ("Classic Movies", "ME GUSTA"),
    ("Comedies", "ME GUSTA"),
    ("Crime TV Shows", "NO ME GUSTA"),
    ("Cult Movies", "ME GUSTA"),
    ("Documentaries", "ME GUSTA"),
    ("Docuseries", "NO ME GUSTA"),
    ("Dramas", "ME GUSTA"),
    ("Faith & Spirituality", "NO ME GUSTA"),
    ("Horror Movies", "ME GUSTA"),
    ("Independent Movies", "NO ME GUSTA"),
    ("International Movies", "ME GUSTA"),
    ("International TV Shows", "ME GUSTA"),
    ("Kids' TV", "NO ME GUSTA"),
    ("Korean TV Shows", "ME GUSTA"),
    ("LGBTQ Movies", "NO ME GUSTA"),
    ("Music & Musicals", "ME GUSTA"),
    ("Reality TV", "NO ME GUSTA"),
    ("Romantic Movies", "ME GUSTA"),
    ("Romantic TV Shows", "NO ME GUSTA"),
    ("Sci-Fi & Fantasy", "ME GUSTA"),
    ("Science & Nature TV", "NO ME GUSTA"),
    ("Spanish-Language TV Shows", "ME GUSTA"),
    ("Sports Movies", "NO ME GUSTA"),
    ("Stand-Up Comedy", "ME GUSTA"),
    ("Stand-Up Comedy & Talk Shows", "NO ME GUSTA"),
    ("TV Action & Adventure", "ME GUSTA"),
    ("TV Comedies", "NO ME GUSTA"),
    ("TV Dramas", "ME GUSTA"),
    ("TV Horror", "NO ME GUSTA"),
    ("TV Mysteries", "ME GUSTA"),
    ("TV Sci-Fi & Fantasy", "NO ME GUSTA"),
    ("TV Shows", "ME GUSTA"),
    ("TV Thrillers", "NO ME GUSTA"),
    ("Teen TV Shows", "ME GUSTA"),
    ("Thrillers", "NO ME GUSTA")
]

# Preprocesamiento de datos: Tokenización y extracción de características
def preprocess(text):
    tokens = word_tokenize(text)
    return {word: True for word in tokens}

# Aplicamos el preprocesamiento a los datos de clasificación
featuresets = [(preprocess(text), sentiment) for text, sentiment in data]

# Entrenamos el clasificador Naive Bayes
train_set, test_set = featuresets[:30], featuresets[30:]
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Crea una instancia de FastAPI
app = FastAPI()
app.title = "Mi aplicación con FastAPI"
app.version = "1.0.0"

# Definie una ruta para la API
@app.get('/', tags=['Home'])
def message():
    return HTMLResponse('<h1>¡Bienvenido a mi aplicación con FastAPI!!!</h1>')

@app.get('/movies', tags=['Movies'])
def get_movies():
    if not movies_list:
        raise HTTPException(status_code=500, detail="No movies data available.")
    return movies_list

@app.get('/movies/{id}', tags=['Movies'])
def get_movie(id: str):
    for item in movies_list:
        if item['id'] == id:
            return item
    return {"Película no encontrada"}

@app.get('/movies/preference/{title}', tags=['Movies'])
def movie_preference(title: str):
    # Busca la película por título
    movie = next((item for item in movies_list if item['title'].lower() == title.lower()), None)
    if not movie:
        return {"detail": "Película no encontrada"}

    # Clasifica según las categorías de la película
    category_text = movie['category']
    features = preprocess(category_text)
    preference = classifier.classify(features)

    return JSONResponse(content={"title": movie['title'], "preference": preference})

@app.get('/movies/', tags=['Movies'])
def get_movies_by_category(category: str):
    return [item for item in movies_list if category.lower() in item['category'].lower()]

@app.post('/movies', tags=['Movies'])
def create_movie(id: str = Body(), title: str = Body(), overview: str = Body(), year: int = Body(), rating: str = Body(), category: str = Body()):
    new_movie = {
        "id": id,
        "title": title,
        "overview": overview,
        "year": year,
        "rating": rating,
        "category": category
    }
    movies_list.append(new_movie)
    return new_movie

@app.put('/movies/{id}', tags=['Movies'])
def update_movie(id: str, title: str = Body(), overview: str = Body(), year: int = Body(), rating: str = Body(), category: str = Body()):
    for item in movies_list:
        if item['id'] == id:
            item.update({
                "title": title,
                "overview": overview,
                "year": year,
                "rating": rating,
                "category": category
            })
            return item
    return {"Película no encontrada"}

@app.delete('/movies/{id}', tags=['Movies'])
def delete_movie(id: str):
    global movies_list
    movies_list = [item for item in movies_list if item['id'] != id]
    return {"Pelicula borrada exitosamente"}