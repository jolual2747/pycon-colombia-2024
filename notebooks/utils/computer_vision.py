import PIL
import os
import io
import urllib
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from typing import Tuple, List, Optional, Dict, Any
import matplotlib.pyplot as plt
from datasets import load_dataset
from datasets.utils.file_utils import get_datasets_user_agent
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class FaceDataset(Dataset):
    """Face Dataset with local images to use in DataLoader."""
    def __init__(self, root_dir: str, transform: bool = True) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.labels = []

        for folder in os.listdir(self.root_dir):
            folder_path = os.path.join(self.root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file)
                    if os.path.isfile(file_path):
                        self.labels.append(file_path)

    def __len__(self) -> int:
        """Get length of dataset.

        Returns:
            int: Length dataset.
        """
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, str]:
        """Get any item by index.

        Args:
            idx (int): Id to retrieve.

        Returns:
            Tuple[np.ndarray, str]: _description_
        """
        img_name = self.labels[idx]
        image = PIL.Image.open(img_name)

        if image.mode != "RGB":
            image = image.convert(mode = "RGB")
                
        image = np.array(image)
        if self.transform:
            image = self.transform_fn(image)
        
        return image, self.labels[idx]

    def transform_fn(self, image: np.ndarray) -> np.ndarray:
        """Transform function to resize every image.

        Args:
            image (np.ndarray): Image to resize

        Returns:
            np.ndarray: Array with shape (512, 512) and dtype np.uint8
        """
        return (resize(image, output_shape = (512, 512)) * 255).astype(np.uint8)
    
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

    fig.suptitle(f'Plotting {num_images} images', fontsize=20)
    plt.tight_layout()
    plt.show()

def fetch_single_image(image_url: str, timeout: bool = None, retries: int = 0) -> Optional[PIL.Image.Image]:
    """Fetch an image from url on internet.

    Args:
        image_url (str): URL to image.
        timeout (bool, optional): Timeout for requests. Defaults to None.
        retries (int, optional): Num. of retries if fails. Defaults to 0.

    Returns:
        Optional[PIL.Image.Image]: Downloaded Image if exists.
    """
    USER_AGENT = get_datasets_user_agent()
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