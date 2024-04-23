import PIL.Image
import transformers
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import io
import urllib
import PIL
from typing import Optional, Dict, Any

USER_AGENT = get_datasets_user_agent()

def fetch_single_image(image_url: str, timeout: bool = None, retries: int = 0) -> Optional[PIL.Image.Image]:
    """Fetch an image from url on internet.

    Args:
        image_url (str): URL to image.
        timeout (bool, optional): Timeout for requests. Defaults to None.
        retries (int, optional): Num. of retries if fails. Defaults to 0.

    Returns:
        Optional[PIL.Image.Image]: Downloaded Image if exists.
    """
    for _ in range(retries + 1):
        try:
            request = urllib.request.Request(
                image_url,
                data=None,
                headers={"user-agent": USER_AGENT},
            )
            with urllib.request.urlopen(request, timeout=timeout) as req:
                image = PIL.Image.open(io.BytesIO(req.read()))
            break
        except Exception:
            image = None
    return image

def fetch_images(batch: dict, num_threads: int, timeout: int = None, retries: int = 0) -> Dict[str, Any]:
    """Fetch images from a Dataset's batch 

    Args:
        batch (dict): Batch of data.
        num_threads (int): Num of threads for multithreading.
        timeout (int, optional): Timeout for requests. Defaults to None.
        retries (int, optional): Num. of retries if fails. Defaults to 0.

    Returns:
        Dict[str, Any]: Batch data with Images.
    """
    fetch_single_image_with_args = partial(fetch_single_image, timeout=timeout, retries=retries)
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        batch["image"] = list(executor.map(fetch_single_image_with_args, batch["imageurl"]))
    return batch

def generate_datasets(dataset_name: str):
    