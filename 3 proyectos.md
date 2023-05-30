# Ciencia-de-datos-ejercicios
ciencia de datos
Análisis de sentimiento de los datos de las redes sociales
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv("tweets.csv")

# Extract the text of the tweets
text = data["text"]

# Create a sentiment lexicon
sentiment_lexicon = pd.read_csv("sentiment_lexicon.csv")

# Calculate the sentiment of each tweet
sentiment = []
for tweet in text:
    sentiment.append(np.mean([sentiment_lexicon[sentiment_lexicon["word"] == word]["sentiment"] for word in tweet.split()]))

# Plot the sentiment of the tweets
plt.hist(sentiment)
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.show()


detección de fraudeimport pandas as pd


importar numpy como np

importar matplotlib.pyplot como plt

# Cargar los datos

datos = pd.read_csv("transacciones.csv")

# Crear un vector de características

características = datos[["cantidad", "tiempo", "comerciante"]]

# Divida los datos en conjuntos de entrenamiento y prueba

X_train, X_test, y_train, y_test = train_test_split(features, data["is_fraud"], test_size=0.25)

# Entrena a un modelo

modelo = LogisticRegression()

modelo.fit(tren_X, tren_y)

# Evaluar el modelo

puntuación = modelo.puntuación(X_test, y_test)

print("Precisión:", puntuación)

# Predecir la tasa de fraude

predicciones = modelo.predecir(X_test)

# Graficar la matriz de confusión

cm = confusion_matrix(y_test, predicciones)

plt.imshow(cm, cmap="Blues")

plt.xlabel("Predicho")

plt.ylabel("Real")

plt.mostrar()
