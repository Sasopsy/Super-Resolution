import torch
import torch.nn as nn
from dataclasses import dataclass
import torch.nn.functional as F
from losses import PerceptualLoss, PerceptualMse
import copy
import math

@dataclass
class ModelArgs:
    upscale_factor: int = 2
    loss: str = 'mse'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Resnet
    num_residual_blocks: int = 4
    # Perceptual loss parameters.
    perceptual_loss_model: str = 'vgg16'
    selected_layers: tuple = (3,8,15,29)
    # With mse.
    perceptual_weight: float = 0.5

# Base SRCNN class

class BaseSRCNN(nn.Module): 
    def __init__(self,
                 args: ModelArgs):
        """Base class for the rest of the srcnn.

        Args:
            args (ModelArgs): Model arguments
            optim (torch.optim): Optimizer object already initiated.
        """
        super(BaseSRCNN,self).__init__()
        self.args = args
        self.loss_dict = {'mse': nn.MSELoss(),
                          'perceptual': PerceptualLoss(
                              selected_layers=self.args.selected_layers,
                              model=self.args.perceptual_loss_model,
                              device=self.args.device,),
                          'perceptual_mse':  PerceptualMse(
                              selected_layers=self.args.selected_layers,
                              model=self.args.perceptual_loss_model,
                              device=self.args.device,
                              perceptual_weight=self.args.perceptual_weight
                          )
                          }
        self.loss = self.loss_dict[copy.deepcopy((self.args.loss))]
        self.to(self.args.device)
        del self.loss_dict
        
    def forward(self,x):
        pass
    
    def train_step(self,
                   downscaled_image: torch.Tensor,
                   original_image: torch.Tensor):
        downscaled_image = downscaled_image.to(self.args.device)
        original_image = original_image.to(self.args.device)
        
        upscaled_image = self(downscaled_image)
        loss = self.loss(original_image,upscaled_image)
        
        return loss,upscaled_image
    
    def save(self,path: str, 
             optim: torch.optim):   
        # Save the important attributes.
        torch.save({
            'model_state_dict': self.state_dict(),
            'optim_state_dict': optim.state_dict(),
            'optim': optim,
            'args': self.args,
        },path)
        
    @classmethod
    def load(cls,path,device=None):
        # Load the state.
        load_dict = torch.load(path,map_location=None)
        # Load the args 
        args = load_dict['args']
        # Create new model object.
        self = cls(args)
        # Load model state dict.
        self.load_state_dict(load_dict['model_state_dict'])
        # Load optim
        optim = load_dict['optim']
        optim.load_state_dict(load_dict['optim_state_dict'])
        
        return self,optim
        
        
# Vanilla SRCNN        

class SRCNN(BaseSRCNN):
    def __init__(self, 
                 args: ModelArgs):
        """_summary_

        Args:
            args (ModelArgs): _description_
            optim (torch.optim): _description_
        """
        super(SRCNN,self).__init__(args)
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), 
            nn.Conv2d(64, 32, kernel_size=5, padding=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.Conv2d(32, 3, kernel_size=5, padding=2),
            nn.Sigmoid())
        
    def forward(self,x: torch.Tensor):
        x = F.interpolate(x,scale_factor=self.args.upscale_factor,mode='bicubic')
        output = self.net(x)
        return output


# SRResNet

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 hidden_channels: int):
        super(ResidualBlock, self).__init__()
        # Used a bottleneck layer as a residual layer with 1x1 convolutions.
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        identity = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += identity  # Element-wise sum: skip connection
        return out

class SubPixelConv(nn.Module):
    def __init__(self, in_channels, upscale_factor=2):
        super(SubPixelConv,self).__init__()
        # Convolution layer to increase the number of channels.
        self.conv = nn.Conv2d(in_channels,in_channels*(upscale_factor**2),kernel_size=3,padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pixle_shuffle = nn.PixelShuffle(upscale_factor)
        
    def forward(self,x):
        x = self.relu(self.conv(x))
        x = self.pixle_shuffle(x)
        return x

class SRResNet(BaseSRCNN):
    def __init__(self,
                 args: ModelArgs):
        super(SRResNet, self).__init__(args)
        
        # Will gradually keep increasing resolution by a factor of 2.
        self.num_sub_pixel_convs = int(math.log2(self.args.upscale_factor))
        
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=2),  
            nn.Sequential(
                *[ResidualBlock(64,32) for _ in range(self.args.num_residual_blocks)]
            ),
            nn.Conv2d(64, 32, kernel_size=3, padding=1), 
            nn.Sequential(
                *[SubPixelConv(32) for _ in range(self.num_sub_pixel_convs)]
            ), 
            nn.Conv2d(32, 3, kernel_size=3, padding=1),  
            nn.Sigmoid())
            
    def forward(self, x):
        output = self.net(x)
        return output
