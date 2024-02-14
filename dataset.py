from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
from torchvision.transforms import ToTensor,Compose,RandomCrop

transform = Compose([
    ToTensor(),
])

class Data(Dataset):
    def __init__(self,image_dir: str,
                 df: pd.DataFrame,
                 downscale_factor: float,
                 crop_size: tuple[int],
                 transform=transform):
        """Dataset creation for super resolution.

        Args:
            image_dir (str): path to image directory.
            df (pd.DataFrame): dataframe associated with directory.
            downscale_factor (float): factor of downscale.
            transform (any): transformations to be applied, Ddfaults to None.
        """
        super(Data,self).__init__()
        self.image_dir = image_dir
        self.df = df
        self.downscale_factor = downscale_factor
        self.transform = transform
        self.random_crop = RandomCrop(size=crop_size,pad_if_needed=True)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # Get image path and open image.
        img_path = f"{self.image_dir}/{self.df.iloc[index]['ImageName']}"
        image = Image.open(img_path).convert('RGB')
        image = self.random_crop(image)
        
        # Downscale the image
        original_size = image.size
        downscaled_size = (int(original_size[0]*self.downscale_factor),
                           int(original_size[1]*self.downscale_factor))
        image_downscaled = image.resize(downscaled_size)
        
        # Self transform
        if self.transform:
            image = self.transform(image)
            image_downscaled = self.transform(image_downscaled)
            
        return image,image_downscaled
            