import os
import pandas as pd 
from torchvision.io import read_image
from torch.utils.data import Dataset

class GNLDataLoader(Dataset):
    def __init__(self, labels_path: str, images_dir: str, transform = None) -> None:
        """
        Creates a dataset given the path to the labels and the image directory

        Parameters:
            - `labels_path`: the path to the `csv` file containing the labels;
            - `images_dir`: the path to the directory with the images;
            - `transform`: states whether a transformation should be applied to the images or not.
        """
        super().__init__()
        self.images_dir = images_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_path)

    def __len__(self) -> int:
        """Returns the length of the dataset
        
        Returns:
            - `length` (`int`): the length of the dataset"""
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple:
        """Get the ith item in the dataset
        
        Parameters:
            - `index`: the index of the image that must be retrieven.
            
        Returns:
            - `image` (`img`): the image in the ith position in the dataset."""
        
        # Get the images path
        images_path = os.path.join(self.images_dir, self.labels.iloc[index, 0])
        image = read_image(images_path) # Can also be done with OpenCV's function cv2.imread()
        label = self.labels.iloc[index, 1]

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        return (image, label)