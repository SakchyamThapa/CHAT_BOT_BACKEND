from typing import List
from sentence_transformers import SentenceTransformer
from loguru import logger

class EmbeddingModelClient:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def get_embedding(self, text: str | List[str]) -> List[float]:
        model = SentenceTransformer(self.model_name)
        embedding = model.encode(text).tolist()
        logger.info(f"Generated embeddings using model: {self.model_name}")
        return embedding
