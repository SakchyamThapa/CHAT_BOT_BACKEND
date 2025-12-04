from time import time
from loguru import logger
from pinecone import Pinecone, ServerlessSpec


class PineconeClient:
    def __init__(self, api_key: str):
        self.client = Pinecone(api_key=api_key, ssl_verify=False)

    def create_index(self, new_index_name: str, dimension: int, metric: str = "cosine"):
        if not self.client.has_index(new_index_name):
            self.client.create_index(
                name=new_index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            logger.info(f"Index {new_index_name} created successfully")
        else:
            logger.info(f"Index {new_index_name} already exists")

        while not self.client.describe_index(new_index_name).status["ready"]:
            time.sleep(1)

    def upsert_vectors(
        self, index_name: str, chunk_text: list, vectors: list, namespace: str
    ):
        records = []
        index = self.client.Index(index_name)
        for i, (d, e) in enumerate(zip(chunk_text, vectors)):
            records.append({"id": f"chunk_{i}", "values": e, "metadata": {"text": d}})

        index.upsert(vectors=records, namespace=namespace)
        logger.info(
            f"Upserted {len(records)} vectors into index: {index_name} in namespace: {namespace}"
        )

    def query(
        self, query_vector: list, index_name: str, name_space: str, top_k: int = 7
    ):
        index = self.client.Index(name=index_name)
        response = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=name_space,
        )
        return response
