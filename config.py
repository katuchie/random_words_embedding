from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
DB_PATH = "./math_knowledge"
COLLECTION_NAME = "math_concepts"


def get_model():
    return SentenceTransformer(MODEL_NAME)