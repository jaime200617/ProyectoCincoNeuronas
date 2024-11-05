# Importamos la libreria de pandas
import pandas as pd

# Definimos la ruta donde se encuentra el dataset y el nombre del dataset.
path = 'DataSets/EnergiaZonasNoInterconectadas.csv'

# Realizamos la carga del dataset y definimos la codificación como ISO8859-1 para permitir caracteres especiales
retail_data = pd.read_csv(path, encoding='iso-8859-1')

# Imprimimos el tipo de variable del retail_data
print(retail_data)

