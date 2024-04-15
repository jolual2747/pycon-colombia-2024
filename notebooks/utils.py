import os
import json
from zipfile import ZipFile
import shutil
from datasets import load_dataset, concatenate_datasets, Dataset
from typing import List

def create_kaggle_credentials() -> None:
    """Creates kaggle.json file with username and key.
    """
    os.system("mkdir -p ~/.kaggle && touch ~/.kaggle/kaggle.json")
    api_token = {"username":os.getenv("kaggle_username"),"key":os.getenv("kaggle_password")}
    
    with open('/teamspace/studios/this_studio/.kaggle/kaggle.json', 'w') as file:
        json.dump(api_token, file)

    os.system("chmod 600 ~/.kaggle/kaggle.json")

def download_kaggle_dataset(dataset_name: str) -> None:
    """Downloads a Kaggle dataset from Kaggle's website.

    Args:
        dataset_name (str): Dataset name to download.
    """
    os.system("kaggle datasets download -d jorgeruizdev/ludwig-music-dataset-moods-and-subgenres")
    with ZipFile("./ludwig-music-dataset-moods-and-subgenres.zip") as zf:
        zf.extractall("../data/unstructured/ludwig-music-dataset-moods-and-subgenres")
    
    os.remove("./ludwig-music-dataset-moods-and-subgenres.zip")

def load_sample_from_genre(genre: str, n_samples: int) -> Dataset:
    """Loads a genre folder and randomly selects n samples.

    Args:
        genre (str): Genre to load.
        n_samples (int): Number of samples to select.

    Returns:
        Dataset: Dataset built from random songs in a genre folder.
    """
    data_path = f"..data/unstructured/ludwig-music-dataset-moods-and-subgenres/mp3/mp3/{genre}"

    return load_dataset(
        cache_dir="../data/unstructured/audiofolder", 
        path = data_path, 
        split = "train"
    ).shuffle(42).select(range(n_samples))

def generate_random_dataset(genres: List[str], n_samples: int) -> Dataset:
    """Generates a random Dataset based on ludwig-music-dataset genres.

    Args:
        genres (List[str]): Genres to use to build the Dataset.
        n_samples (int): Number of samples for each genre.

    Returns:
        Dataset: Dataset built from random songs selected from the genres list.
    """
    ds = None
    for genre in genres:
        print(f"Loading {genre}...")
        if ds is None:
            ds = load_sample_from_genre(genre, n_samples)
        else:
            ds_aux = load_sample_from_genre(genre, n_samples)
            ds = concatenate_datasets([ds_aux, ds])

    del ds_aux
    return ds