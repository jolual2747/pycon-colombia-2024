import PIL
import os
import numpy as np
from skimage.transform import resize
from torch.utils.data import Dataset
from typing import Tuple

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