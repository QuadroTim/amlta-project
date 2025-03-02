from typing import NamedTuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_qdrant.qdrant import QdrantVectorStore
from qdrant_client.client_base import QdrantBase
from qdrant_client.models import models
from sentence_transformers import CrossEncoder
from tqdm.auto import tqdm

from amlta.app.rag.client import get_qdrant_client
from amlta.app.rag.loaders import (
    HasLengthLoader,
    YamlGlossaryLoader,
    YamlProcessLoader,
)


class PrefixedHuggingFaceEmbeddings(HuggingFaceEmbeddings):
    passage_prefix: str = "passage"
    query_prefix: str = "query"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return super().embed_documents(
            [f"{self.passage_prefix}: {text}" for text in texts]
        )

    def embed_query(self, text: str) -> list[float]:
        return super().embed_query(f"{self.query_prefix}: {text}")


def get_or_create_collection(client: QdrantBase, name: str):
    if not client.collection_exists(name):
        client.create_collection(
            name,
            vectors_config={
                # intfloat/multilingual-e5-large
                "dense": models.VectorParams(
                    size=1024, distance=models.Distance.COSINE
                ),
            },
            # sparse_vectors_config={
            #     "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            # },
        )

    return client.get_collection(name)


class Collections(NamedTuple):
    glossary: QdrantVectorStore
    processes: QdrantVectorStore

    reranker: CrossEncoder


def load_documents(
    collection: QdrantVectorStore, loader: HasLengthLoader, chunk_size: int = 64
):
    stored_uuids = {record.id for record in iter_collection(collection)}

    if len(stored_uuids) == len(loader):
        return

    pbar = tqdm(
        desc=f"Adding documents ({collection.collection_name})", total=len(loader)
    )
    docs_gen = loader.lazy_load()

    def load_chunk():
        i = 0
        chunk = []

        for doc in docs_gen:
            pbar.update(1)
            if doc.id in stored_uuids:
                continue

            chunk.append(doc)
            i += 1

            if i >= chunk_size:
                break

        return chunk

    while chunk := load_chunk():
        collection.add_documents(chunk)


def get_collections(client: QdrantBase):
    get_or_create_collection(client, "glossary")
    get_or_create_collection(client, "processes")

    # cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

    embedding = PrefixedHuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large"
    )

    collections = Collections(
        glossary=QdrantVectorStore(
            client,  # type: ignore
            "glossary",
            vector_name="dense",
            embedding=embedding,
            sparse_vector_name="bm25",
            sparse_embedding=FastEmbedSparse(),
            retrieval_mode=RetrievalMode.HYBRID,
        ),
        processes=QdrantVectorStore(
            client,  # type: ignore
            "processes",
            vector_name="dense",
            embedding=embedding,
            sparse_vector_name="bm25",
            sparse_embedding=FastEmbedSparse(),
            retrieval_mode=RetrievalMode.HYBRID,
        ),
        reranker=cross_encoder,
    )

    # glossary_loader = MarkdownGlossaryLoader()
    # process_loader = MarkdownProcessLoader()
    glossary_loader = YamlGlossaryLoader()
    process_loader = YamlProcessLoader()
    load_documents(collections.glossary, glossary_loader)
    load_documents(collections.processes, process_loader)

    return collections


def iter_collection(store: QdrantVectorStore):
    records, offset = store.client.scroll(store.collection_name, limit=1024)

    yield from records

    while offset:
        records, offset = store.client.scroll(
            store.collection_name, limit=1024, offset=offset
        )
        yield from records


def main():
    collections = get_collections(get_qdrant_client())
    process_store = collections.processes

    while query := input("Enter a query: "):
        results = process_store.similarity_search(query, k=5)
        for result in results:
            print(result.page_content[:100])
            print("-" * 80)
            print()


if __name__ == "__main__":
    main()
