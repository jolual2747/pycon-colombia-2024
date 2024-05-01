from qdrant_client import QdrantClient
from transformers import ViTImageProcessor, ViTModel
from typing import Tuple
import streamlit as st
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
from ast import literal_eval
import re

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
    model = ViTModel.from_pretrained("./assets/embedding_model")
    return processor, model

@st.cache_data
def get_products_data() -> pd.DataFrame:
    """Load products data.

    Returns:
        pd.DataFrame: Products with metadata and images.
    """
    df = pd.read_csv("./data/structured/products.csv")
    df["image"] = df["image"].apply(literal_eval)
    return df


def plot_images_grid(image_list: List[np.ndarray], titles: List[str] = None) -> None:
    """Plot images from a list of arrays in a grid layout.

    Args:
        image_list (List[np.ndarray]): List of images as arrays.
        titles (List[str], optional): Titles to map to every image. Defaults to None.
    """
    num_images = len(image_list)
    num_columns = 4
    num_rows = (num_images + num_columns - 1) // num_columns  # Calcular el número de filas necesarias

    fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 4 * num_rows))  # Crear la cuadrícula de subgráficos

    # Ocultar los ejes y los subgráficos vacíos
    for i in range(num_rows):
        for j in range(num_columns):
            index = i * num_columns + j
            if index < num_images:
                axes[i, j].imshow(image_list[index])
                axes[i, j].axis('off')
                if titles is not None and index < len(titles):
                    title = result = re.sub(r'(?:\s+\S+){3}\s+', lambda m: m.group()[:-1] + '\n', titles[index], count=1)
                    axes[i, j].set_title(title)
            else:
                axes[i, j].axis('off')  # Ocultar los ejes para los subgráficos vacíos

    plt.tight_layout()
    st.pyplot(fig)
