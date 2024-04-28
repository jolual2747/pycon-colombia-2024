from qdrant_client import QdrantClient
from transformers import ViTImageProcessor, ViTModel
from typing import Tuple
import streamlit as st

def get_client(host: str, port: int) -> QdrantClient:
    """Get a QdrantClient to manage vector databases. 

    Args:
        host (str): Host name of Qdrant service. 
        port (int): Port of the Qdrant service.

    Returns:
        QdrantClient: QdrantClient
    """
    return QdrantClient(host=host, port=port)

@st.cache_resource
def load_model() -> Tuple[ViTImageProcessor, ViTModel]:
    """Load ViT model and Image processor.

    Returns:
        Tuple[ViTImageProcessor, ViTModel]: ViT model and Image processor
    """
    model_id = 'jolual2747/vit-clothes-classification'
    processor = ViTImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id)
    return model, processor