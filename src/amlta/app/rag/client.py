import streamlit as st
from qdrant_client.local.qdrant_local import QdrantLocal

from amlta.config import config


class QdrantLocalClient(QdrantLocal):
    def __init__(self, location: str | None = None, **kwargs):
        super().__init__(location=location or str(config.data_dir / "qdrant"), **kwargs)


@st.cache_resource
def get_qdrant_client(location: str | None = None):
    return QdrantLocalClient(location=location)
