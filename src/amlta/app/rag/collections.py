from typing import NamedTuple

from langchain_community.embeddings import FakeEmbeddings
from langchain_qdrant import RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client.client_base import QdrantBase
from qdrant_client.models import models


def get_or_create_collection(client: QdrantBase, name: str):
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={
                # TODO: model
                # paraphrase-multilingual-MiniLM-L12-v2
                "fake": models.VectorParams(
                    size=384,
                    distance=models.Distance.COSINE,
                ),
                # "colbertv2.0": models.VectorParams(
                #     size=len(late_interaction_embeddings[0][0]),
                #     distance=models.Distance.COSINE,
                #     multivector_config=models.MultiVectorConfig(
                #         comparator=models.MultiVectorComparator.MAX_SIM,
                #     ),
                # ),
            },
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
        )

    return client.get_collection(name)


class Collections(NamedTuple):
    glossary: QdrantVectorStore
    processes: QdrantVectorStore


def get_collections(client: QdrantBase):
    get_or_create_collection(client, "glossary")
    get_or_create_collection(client, "processes")

    return Collections(
        glossary=QdrantVectorStore(
            client,  # type: ignore
            "glossary",
            vector_name="fake",
            embedding=FakeEmbeddings(size=384),
            sparse_vector_name="bm25",
            sparse_embedding=FastEmbedSparse(),
            retrieval_mode=RetrievalMode.HYBRID,
        ),
        processes=QdrantVectorStore(
            client,  # type: ignore
            "processes",
            vector_name="fake",
            embedding=FakeEmbeddings(size=384),
            sparse_vector_name="bm25",
            sparse_embedding=FastEmbedSparse(),
            retrieval_mode=RetrievalMode.HYBRID,
        ),
    )


def iter_collection(store: QdrantVectorStore):
    records, offset = store.client.scroll(store.collection_name, limit=1024)

    yield from records

    while offset:
        records, offset = store.client.scroll(
            store.collection_name, limit=1024, offset=offset
        )
        yield from records
