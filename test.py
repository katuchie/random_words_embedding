from embedder import text_to_embedding
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import pickle

words =  [
    "математика", "теорема Коши", "мама", "вода", "Виктория Казей",
    "МГТУ им.Баумна", "mathematics", "любовь", "Виктор Цой"
]

vec_words = [text_to_embedding(w) for w in words]
vec_words =  np.array(vec_words)

#уменьшим размерность
pca = PCA(n_components=2)
vectors_2d = pca.fit_transform(vec_words)

with open('vectors_2d.pkl', 'wb') as f:
    pickle.dump((words, vectors_2d), f)
