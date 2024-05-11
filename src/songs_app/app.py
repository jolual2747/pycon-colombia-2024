import PIL.Image
import streamlit as st
import PIL
import numpy as np
from io import BytesIO
import torch
import time
from utils import (
    extract_audio_from_youtube_video,
    get_client,
    tmp_folder
)

collection = "music_collection3"

def main() -> None:
    """
    Main of the Streamlit app. 
    """
    st.title("Find cool songs.")
    st.write("Enter a URL YouTube video and find songs similar to your favorite ones")

    with st.sidebar:
        st.title("Upload your URL!")
        url = st.text_input("Copy and paste an URL from YouTube")
        if url and st.button("Search"):
            st.session_state.url = url

        
    if url and "url" in st.session_state:
        with st.spinner("Please wait, downloading video..."):
            extract_audio_from_youtube_video(url, "audio.mp3")


if __name__ == '__main__':
    main()