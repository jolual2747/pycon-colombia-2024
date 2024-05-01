import PIL.Image
import streamlit as st
import PIL
import numpy as np
from io import BytesIO
import torch
import time
from utils import (
    get_client, 
    plot_images_grid, 
    load_model, 
    get_products_data
)

collection = "products2"

def main() -> None:
    """
    Main of the Streamlit app. 
    """
    st.title("Search a product by an image as query.")
    st.write("Upload your product!")
    client = get_client("localhost", 6333)
    processor, model = load_model()
    ds_pandas = get_products_data()

    with st.sidebar:
        st.title("Upload your product!")
        uploaded_file = st.file_uploader("Upload a pic", type=["png", "jpeg"])
        if uploaded_file and st.button("Search"):
            st.image(uploaded_file)
            st.session_state.uploaded_img = PIL.Image.open(uploaded_file)

    if uploaded_file and "uploaded_img" in st.session_state:
        inputs = processor(images=st.session_state.uploaded_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.pooler_output
    
        results = client.search(
            collection_name=collection,
            query_vector=embedding.detach().cpu().numpy().tolist()[0],
            score_threshold=0.01,
            limit=10
        )

        if len(results) > 0:
            res = [result.id for result in results]
            descriptions = [result.payload["title"] for result in results]
            similar_images = []
            start = time.time()
            for row in ds_pandas.iloc[res,]["image"]:
                similar_images.append(np.array(PIL.Image.open(BytesIO(row["bytes"]))))
            # similar_images = load_images_parallel(ds_pandas.iloc[res,])
            plot_images_grid(similar_images, titles=descriptions)
            end = time.time()
            print(f"Time {end-start}")
        
        else:
            st.error("Not found similar products. Try with another!")
            
        

if __name__ == '__main__':
    main()