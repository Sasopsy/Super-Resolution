import models
import torch
import os
from torch.utils.data import DataLoader
import pandas as pd
from dataclasses import dataclass
from dataset import Data
from tqdm import tqdm
from torcheval.metrics.functional import peak_signal_noise_ratio


@dataclass
class TrainConfigs:
    learning_rate: float = 1e-4
    epochs: int = 10
    batch_size: int = 1
    save_every: int = 2
    crop_size: int = (300,300)
    
    save_path: str = 'train_files'
    images_path: str = 'images'
    train_csv_path: str = 'train.csv'
    val_csv_path: str = 'val.csv'
    
    optimizer: torch.optim = torch.optim.Adam
    
    def __post_init__(self):
        self.train_df = pd.read_csv(self.train_csv_path)
        self.val_df = pd.read_csv(self.val_csv_path)
        
        
class Trainer(object):

    def __init__(self,
                 configs: TrainConfigs,
                 model: models.BaseSRCNN,
                 optim: torch.optim):
        """Trainer object.

        Args:
            configs (TrainConfigs): configurations of training.
            model (models.BaseSRCNN): model to trained.
            optim (torch.optim): uninitiated optimizer object.
        """
        self.configs = configs
        self.model = model
        
        # Create train and val dataset objects.
        self.train_dataset = Data(self.configs.images_path,
                                  self.configs.train_df,
                                  1/self.model.args.upscale_factor,
                                  self.configs.crop_size)
        self.val_dataset = Data(self.configs.images_path,
                                 self.configs.val_df,
                                 1/self.model.args.upscale_factor,
                                 self.configs.crop_size)
        
        # Dataloaders
        self.train_loader = DataLoader(self.train_dataset,
                                       self.configs.batch_size,
                                       shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,
                                      self.configs.batch_size,
                                      shuffle=False)
        
        # Create optimizer object.
        self.optim = optim(self.model.parameters(),lr = self.configs.learning_rate)
        
        # Create history for plotting later.
        self.train_history = []
        self.val_history = []
        self.psnr_train = []
        self.psnr_val = []
        
        # Epoch tracker
        self.epoch = 0
        
    def train(self):
        """Train function for our model.
        """
        for epoch in range(self.configs.epochs):
            loop = tqdm(enumerate(self.train_loader))
            
            # Train
            train_loss = 0.0
            train_psnr = 0.0
            for i, (image,image_downscaled) in loop:
                loss,upscaled_image = self.model.train_step(image_downscaled,image)
                self.model.zero_grad()
                loss.backward()
                self.optim.step()
                # Calculate psnr
                psnr = peak_signal_noise_ratio(image,upscaled_image.cpu(),1.0)
                # Postfix for tqdm object.
                loop.set_postfix(loss=loss.cpu().item(),psnr=psnr.cpu().item())
                
                # Scaling loss and psnr with batch size to calculate accurate loss later.
                train_loss += loss.cpu().item()*image.shape[0]
                train_psnr += psnr.cpu().item()*image.shape[0]
            # Normalize by total number of images.
            train_loss = train_loss/len(self.train_dataset)
            train_psnr = train_psnr/len(self.train_dataset)

            # Append loss and psnr
            self.train_history.append(train_loss)
            self.psnr_train.append(train_psnr)
            
            # Val
            val_loss = 0.0
            val_psnr = 0.0
            with torch.no_grad():
                for i,(image,image_downscaled) in enumerate(self.val_loader):
                    loss,upscaled_image = self.model.train_step(image_downscaled,image)
                    # Update loss and psnr
                    val_loss += loss.cpu().item()*image.shape[0]
                    val_psnr += peak_signal_noise_ratio(image,upscaled_image.cpu(),1.0).cpu().item()*image.shape[0]
            # Normalize
            val_loss = val_loss/len(self.val_dataset)
            val_psnr = val_psnr/len(self.val_dataset)
            
            # Append loss and psnr
            self.val_history.append(val_loss)
            self.psnr_val.append(val_psnr)
            
            # Checkpointing
            if self.epoch%self.configs.save_every == 0:
                self.save()
                
            # Display val loss and val psnr.
            print(f"Val loss: {val_loss} | Val psnr: {val_psnr}")
                
            # Update epoch number
            self.epoch += 1
            
                
    def save(self):
        # Create directory.
        if not os.path.exists(self.configs.save_path):
            os.mkdir(self.configs.save_path)
        
        # Separate directories for different epoch path
        dir = os.path.join(self.configs.save_path,f'epoch_{self.epoch}')
        if not os.path.exists(dir):
            os.mkdir(dir)
            # Save the model first.
            self.model.save(os.path.join(dir,f'model.pt'),
                            self.optim)
            torch.save({
                'configs': self.configs.__dict__,
                'history': [self.train_history,
                            self.val_history,
                            self.psnr_train,
                            self.psnr_val],
                'epoch': self.epoch
                },os.path.join(dir,'trainer.pkl'))
        else:
            print("This version already exists.")
    
    @classmethod
    def load(cls, path: str, epoch: int,type,device=torch.device('cpu')):
        """Loads trainer with the model.

        Args:
            path (str): path to directory.
            epoch (int): epoch number of the model.
            type (any): type of srcnn used. (uninitiated object)
        """
        checkpoint_path = os.path.join(path,f'epoch_{epoch}')
        model,optim = type.load(os.path.join(checkpoint_path,f'model.pt'),device=device)
        load_dict = torch.load(os.path.join(checkpoint_path,f'trainer.pkl'))
        configs_dict = load_dict['configs']
        
        # Create new configs.
        configs = TrainConfigs()
        
        # Set items in the configs dict into our new configs object.
        for attribute,value in configs_dict.items():
            setattr(configs, attribute, value)
        
        # Nested history list.
        his = load_dict['history']
        
        # Create new class instance.
        self = cls(configs,model,torch.optim.Adam)  # Dummy optimizer.
        
        # Put in real optimizer
        self.optim = optim
        
        # Extract loss and metric histories.
        self.train_history = his[0]
        self.val_history = his[1]
        self.psnr_train = his[2]
        self.psnr_val = his[3]
        
        # Get epoch count
        self.epoch = load_dict['epoch']
        
        return self
            
        