import numpy as np
from config import get_model

_model = None

def get_embedder():
    global _model
    if _model is None:
        _model = get_model()
    return _model

def text_to_embedding(text: str, normalize: bool = True):
    model = get_embedder()
    embedding = model.encode(text) # превращаем текст в эмбеддинг
    if normalize:
        embedding = embedding / np.linalg.norm(embedding) # исходный_вектор  / вычисляет длину вектора
    return embedding.tolist()