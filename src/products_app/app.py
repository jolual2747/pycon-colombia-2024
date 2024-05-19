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

hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden; }
    footer {visibility: hidden; }
    .stDeployButton {
            visibility: hidden;
        }
    </style>
"""

def start_over_with_new_search():
    """
    Deletes objects from Streamlit's session.
    """
    # Delete uploaded images
    del st.session_state.uploaded_img
    # display message to user
    st.info('Please upload a new image to continue!')

def main() -> None:
    """
    Main of the Streamlit app. 
    """
    st.set_page_config(layout="wide")
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title("A Fancy Fashion Store 👘")
    st.write("Search a product by an image as query 🔎")
    client = get_client("localhost", 6333)
    processor, model = load_model()
    ds_pandas = get_products_data()

    with st.sidebar:
        st.title("Upload your product!")
        uploaded_file = st.file_uploader("Upload a pic", type=["png", "jpeg"])
        if uploaded_file and st.button("Search"):
            st.session_state.uploaded_img = PIL.Image.open(uploaded_file).convert(mode = "RGB")
            st.button('Clear results', on_click=start_over_with_new_search, key='new_search')
            st.subheader(f"Uploaded image ✅:")
            st.image(uploaded_file)

    if uploaded_file and "uploaded_img" in st.session_state:
        inputs = processor(images=st.session_state.uploaded_img, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.pooler_output
        
        st.divider() 

        #with st.spinner("Please wait, we are looking our most similar products..."):
        with st.status("Please wait, we are looking our most similar products...", expanded=True) as status:
            st.write("Searching for products...")
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
                st.write("Showing results...")
                st.success("""Results:""")
                plot_images_grid(similar_images, titles=descriptions)
                end = time.time()
                print(f"Time {end-start}")
                status.update(label="Results found!", state="complete", expanded=True)
            
            else:
                st.error("Not found similar products. Try with another!")


if __name__ == '__main__':
    main()