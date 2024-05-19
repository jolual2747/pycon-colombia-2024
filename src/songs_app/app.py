import os
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from IPython.display import Audio
from utils import (
    extract_audio_from_youtube_video,
    get_client,
    tmp_folder,
    load_model,
    read_audio_as_array,
    get_one_embedding,
    get_youtube_video_info
)

collection = "music_collection3"

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
    del st.session_state.url
    # display message to user
    st.info('Please upload a new url to continue!')


def main() -> None:
    """
    Main of the Streamlit app. 
    """
    # Title and resources
    st.set_page_config(layout="wide")
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.title("Find Cool Songs. 🎵🎶😎")
    st.write("Enter a URL YouTube video and find songs similar to your favorite ones")
    model = load_model()
    client = get_client("localhost", 6333)

    # Input for URL
    with st.sidebar:
        st.title("Upload your URL!")
        url = st.text_input("Copy and paste an URL from YouTube")
        if url and st.button("Search"):
            st.session_state.url = url
            st.button('Clear results', on_click=start_over_with_new_search, key='new_search')
            uploaded_video, thumbnail = get_youtube_video_info(st.session_state.url)
            st.subheader(f"Uploaded video ✅:\n{uploaded_video}")
            st.image(thumbnail)

    # Download video, embedding and search similar songs
    if url and "url" in st.session_state:
        with st.spinner("Please wait, downloading video..."):
            extract_audio_from_youtube_video(url, "audio.mp3")

        # Compute embedding
        frame_rate, audio_array = read_audio_as_array(os.path.join(tmp_folder, "audio.mp3"))
        embedding = get_one_embedding(model, audio_array)
        embedding = embedding[0].tolist()

        # Search similar songs
        results = client.search(
            collection_name=collection,
            query_vector=embedding,
            limit=10
        )

        # Display results
        if len(results) > 0:
            st.markdown("---")
            songs_names = [result.payload for result in results]
            for idx, song in enumerate(songs_names):
                song["similarity"] = round(results[idx].score, 2)               

            st.markdown("""## Results:""")            

            row1 = st.columns(2)
            row2 = st.columns(2)
            row3 = st.columns(2)
            row4 = st.columns(2)
            row5 = st.columns(2)

            for idx, col in enumerate(row1 + row2 + row3 + row4 + row5):
                tile = col.container(height=220)
                tile.write(f"Result {idx + 1}")
                tile.dataframe(pd.DataFrame.from_records([songs_names[idx]]).drop(columns = ["urls"]), hide_index = True)
                tile.audio(songs_names[idx]["urls"][1:])

            # res = [result.id for result in results]
            # songs_names = [result.payload for result in results]
            # entries_to_remove = ('subgenres', 'urls')
            # for k in entries_to_remove:
            #     for song in songs_names:
            #         song.pop(k, None)
            # st.write(songs_names)
            # for song in songs_names:
            #     st.dataframe(pd.DataFrame.from_records([song]).drop(columns = ["urls"]))
            #     st.audio(song["urls"][1:])

            # data = pd.DataFrame.from_records(songs_names)
            # st.dataframe(data)
        
        else:
            st.error("Not found similar songs. Try with another!")

if __name__ == '__main__':
    main()