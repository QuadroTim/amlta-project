from amlta.app.rag.client import get_qdrant_client
from amlta.app.rag.collections import get_collections

qdrant_client = get_qdrant_client("data/qdrant-yaml")
collections = get_collections(qdrant_client)
