import streamlit as st
from utils import get_client
import PIL


def main() -> None:
    """
    Main of the Streamlit app. 
    """
    st.title("Search a product by an image as query.")
    st.write("Upload your product!")
    client = get_client("localhost", 6333)

    with st.sidebar:
        st.title("Upload your product!")
        uploaded_file = st.file_uploader("Upload a pic", type=["png"])
        if uploaded_file and st.button("Search"):
            st.image(uploaded_file)
            st.write(type(uploaded_file))


if __name__ == '__main__':
    main()