import os
import PIL.Image
import numpy as np
import torch
import PIL
from io import BytesIO
from typing import Any, Tuple, Optional
from pytube import YouTube
import requests
from bs4 import BeautifulSoup
from pydub import AudioSegment
from qdrant_client import QdrantClient
from panns_inference import AudioTagging
import streamlit as st

tmp_folder = os.path.join(os.getcwd(), "src", "songs_app", "tmp")

def extract_audio_from_youtube_video(url_video: str, file_name: str):
    """Download 

    Args:
        url_video (str): URL of Youtube Video.
        file_name (str): File name to store in tmp folder. Must end like .mp3
    """
    # Download video
    mp4_file_name = os.path.join(tmp_folder, "video.mp4")
    try:
        yt = YouTube(url_video)
        audio = yt.streams.filter(only_audio=True).first()
        audio.download(filename=mp4_file_name)
    except:
        raise ReferenceError("The YouTube URL could be broken or not exists!")

    # Extract audio
    audio = AudioSegment.from_file(mp4_file_name, format="mp4")

    # Extract 30 seconds
    inicio_ms = 30000  # Seconds 30 (3 * 1000)
    fin_ms = 60000    # Seconds 60 (20 * 1000)
    audio = audio[inicio_ms:fin_ms]
    audio = audio.set_channels(1)
    audio = audio.set_frame_rate(44100)

    # Store MP3 file
    audio.export(os.path.join(tmp_folder, file_name), format="mp3")

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
def load_model() -> Any:
    """Load PANNs model embedding.

    Returns:
        Any: Model embedding.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    at = AudioTagging(checkpoint_path=None, device=device)
    return at

def read_audio_as_array(file_path: str, normalized: bool = True) -> Tuple[int, np.ndarray]:
    """MP3 to numpy array

    Args:
        file_path (str): Path to .mp3 file.
        normalized (bool, optional): To normalize data. Defaults to True.

    Returns:
        Tuple[int, np.ndarray]: Frame rate and audio array.
    """
    a = AudioSegment.from_mp3(file_path)
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y    

def get_one_embedding(model: Any, audio_array: np.ndarray) -> np.ndarray:
    """Generate embeddings from an audio array.

    Args:
        model (Any): PANNs model embedding.
        audio_array (np.ndarray): Numpy array representation of audio .mp3 file.

    Returns:
        np.ndarray: Embedding.
    """
    array = torch.tensor(audio_array, dtype=torch.float64).unsqueeze(0)
    inputs = torch.nn.utils.rnn.pad_sequence(array, batch_first=True, padding_value=0).type(torch.FloatTensor)
    with torch.no_grad():
        _, embedding = model.inference(inputs)
    return embedding

def get_youtube_video_info(url: str) -> Tuple[str, Optional[PIL.Image.Image]]:
    """Get YouTube's video title and thumbnail.

    Args:
        url (str): URL of Youtube Video. 

    Returns:
        Tuple[str, Optional[PIL.Image.Image]]: Title and thumbnail.
    """
    r = requests.get(url)
    soup = BeautifulSoup(r.text)

    link = soup.find_all(name="title")[0]
    title = link.text

    # Find og:image metadata
    meta_image = soup.find('meta', property='og:image')

    # Get thumbnail image
    thumbnail = None
    if meta_image:
        thumbnail_url = meta_image['content']
        print("Miniatura del video:", thumbnail_url)

        # Download image
        response_image = requests.get(thumbnail_url)

        # Convert to PIL object
        thumbnail = PIL.Image.open(BytesIO(response_image.content))

    return title, thumbnail