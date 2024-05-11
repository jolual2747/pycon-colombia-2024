import os
from pytube import YouTube
from pydub import AudioSegment
from qdrant_client import QdrantClient

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
