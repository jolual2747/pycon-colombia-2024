from qdrant_client import QdrantClient
from transformers import ViTImageProcessor, ViTModel
from typing import Tuple
import streamlit as st
import matplotlib.pyplot as plt
from typing import List
import numpy as np
import pandas as pd
from ast import literal_eval

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

def plot_images_horizontal(image_list: List[np.ndarray], titles: List[str] = None) -> None:
    """Plot images from a list of arrays.

    Args:
        image_list (List[np.ndarray]): List of images as arrays.
        titles (List[str], optional): Titles to map to every image. Defaults to None.
    """
    num_images = len(image_list)

    if num_images != len(titles):
        raise ValueError(f"Number of images is different from number of titles")
    
    # Create fig and axes
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 4, 4))
    
    # Hide axis
    for ax in axes:
        ax.axis('off')
    
    # Plot images
    for i, (image, ax) in enumerate(zip(image_list, axes)):
        ax.imshow(image)  
        if titles is not None:
            ax.set_title(f"\n{titles[i]}")    

    fig.suptitle(f'Similar {num_images} products', fontsize=20)
    plt.tight_layout()

    st.pyplot(fig)