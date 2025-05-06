from flask import Flask, request, render_template, send_file
import pandas as pd
import numpy as np
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import Word2Vec
import os

# Inicializar Flask
app_clasificacion = Flask(__name__)
import zipfile

# Descomprimir si no está ya
if not os.path.exists("modelo_rf.pkl"):
    with zipfile.ZipFile("modelo_rf.zip", "r") as zip_ref:
        zip_ref.extractall(".")

rf_model = joblib.load("modelo_rf.pkl")
scaler = joblib.load("scaler.pkl")
word2vec_model = Word2Vec.load("word2vec.model")

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('spanish'))
palabras_excepcion = {"sin", "fuera", "estado", "no"}
lemmatizer = WordNetLemmatizer()

# Función para limpiar texto
def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = re.sub(f"[{string.punctuation}]", " ", texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    palabras = texto.split()
    palabras = [word for word in palabras if word not in stop_words or word in palabras_excepcion]
    palabras = [lemmatizer.lemmatize(word) for word in palabras]
    return " ".join(palabras)

# Función para obtener embedding
def get_embedding(texto):
    palabras = texto.split()
    vector = np.zeros(200)
    count = 0
    for palabra in palabras:
        if palabra in word2vec_model.wv:
            vector += word2vec_model.wv[palabra]
            count += 1
    return vector / count if count > 0 else vector

# Función para detectar la fila con encabezados
def encontrar_fila_encabezado(archivo, columnas, max_rows=20):
    encabezado_raw = pd.read_excel(archivo, header=None, nrows=max_rows)
    for i in range(len(encabezado_raw)):
        fila = encabezado_raw.iloc[i].astype(str).str.strip().str.lower()
        if all(col.lower() in fila.values for col in columnas):
            return i
    return None

@app_clasificacion.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        archivo = request.files["archivo"]
        columnas_necesarias = ["Descripción"]
        encabezado = encontrar_fila_encabezado(archivo, columnas_necesarias)

        if encabezado is None:
            return "No se encontraron las columnas requeridas. Revisa que esté la columna 'Descripción'."

        df = pd.read_excel(archivo, skiprows=encabezado)

        # Limpiar filas vacías
        df = df.dropna(subset=["Descripción"])
        df = df[df["Descripción"].astype(str).str.strip() != ""]

        # Limpiar texto
        df["Descripción_limpia"] = df["Descripción"].apply(limpiar_texto)

        # Vectorizar y clasificar
        resultados = []
        for _, fila in df.iterrows():
            desc_limpia = fila["Descripción_limpia"]
            emb = get_embedding(desc_limpia)
            emb_s = scaler.transform([emb])
            pred = rf_model.predict(emb_s)[0]
            resultados.append(pred)

        df["Resultado"] = resultados

        # Guardar archivo de salida
        salida = "Resultados_Clasificacion_W2V.xlsx"
        df.to_excel(salida, index=False)

        return send_file(salida, as_attachment=True)

    return render_template("index.html")
