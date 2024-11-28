import nltk 
import random 

data = [
    ("TIGRE", "MAMIFERO"),
    ("SERPIENTE", "REPTIL"),
    ("AGUILA", "AVE"),
    ("TIBURON", "PEZ"),
    ("RANA", "ANFIBIO"),
    ("ELEFANTE", "MAMIFERO"),
    ("TORTUGA", "REPTIL"),
    ("COLIBRI", "AVE"),
    ("SALMON", "PEZ"),
    ("SALAMANDRA", "ANFIBIO"),
    ("GATO", "MAMIFERO"),
    ("IGUANA", "REPTIL"),
    ("PALOMA", "AVE"),
    ("PEZ", "PEZ"),
    ("SAPO", "ANFIBIO"),
    ("PERRO", "MAMIFERO"),
    ("COBRA", "REPTIL"),
    ("PINGUINO", "AVE"),
    ("BACALAO", "PEZ")
]
# Preprocesamiento de datos: Tokenización y extracción de características
def preprocess(text):
    tokens = nltk.word_tokenize(text)
    return {word: True for word in tokens}

# Aplicamos el preprocesamiento a los datos
featuresets = [(preprocess(text), sentiment) for text, sentiment in data]

# Dividimos los datos en conjuntos de entrenamiento y prueba
train_set, test_set = featuresets[:16], featuresets[16:]

# Entrenamos un clasificador utilizando Naive Bayes
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Evaluamos el clasificador en el conjunto de prueba
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Clasificamos un nuevo texto
while True:
    # Capturamos el texto de entrada desde la terminal
    new_text = input("Please enter the text to classify (type 'salir' to exit): ").upper()
    
    # Verificamos si el usuario quiere salir
    if new_text.lower() == "salir":
        print("Exiting the program.")
        break
    
    # Preprocesamos el texto y lo clasificamos
    new_text_features = preprocess(new_text)
    predicted_label = classifier.classify(new_text_features)
    
    # Imprimimos el resultado
    print("Predicted sentiment:", predicted_label)