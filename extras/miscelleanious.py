import torch
import torch.nn as nn
from torchvision.models import vgg19,vgg16,VGG16_Weights,VGG19_Weights

def fetch_model(model: str):
    """Fetches model to be used for perceptual loss."""
    # Dictionary of models.
    models_ = {'vgg19':(VGG19_Weights,vgg19),
               'vgg16':(VGG16_Weights,vgg16)}
    # Fetch and load the model.
    model_ = models_[model][1](weights=models_[model][0].DEFAULT).features  # Will only return feature extractor layers in our models.
    return model_

class PerceptualLoss(nn.Module):
    """Object that calculates perceptual loss using a pre-trained classification model.
    
    Args:
        selected_layers (list[int]): list of the index of layers to be used.
        model (str): model to be used for perceptual loss.
        weights (list[float]): list of weights of outputs.
        alpha (float): weight of reconstruction of loss.
        beta (float): weight of style loss.
    """
    def __init__(self, selected_layers: list[int],
                 model: str,
                 weights: list[float],
                 alpha: float,
                 beta: float,
                 device: str) -> None:
        assert len(selected_layers) == len(weights), "There must be same number of layers outputs and weights."
        super(PerceptualLoss,self).__init__()
        self.model = fetch_model(model).eval().to(device)
        self.selected_layers = sorted(selected_layers)
        self.weights = weights
        self.alpha = alpha
        self.beta = beta
        
        # Freeze the vgg parameters
        for params in self.model.parameters():
            params.requires_grad = False
        
        # List of slices through which our image will pass.
        self.slices = nn.ModuleList([nn.Sequential() for _ in range(len(selected_layers))])
        
        # Keep track of the index of start layer of every slice.
        start_layer = 0
        for i,layer in enumerate(selected_layers):
            for x in range(start_layer,layer+1): 
                self.slices[i].add_module(str(x),self.model[x])
            start_layer = layer+1
            
    def compute_gram_matrix(self,image_unrolled: torch.Tensor):
        # Computes gram matrix of a tensor.
        return torch.matmul(image_unrolled,
                            image_unrolled.transpose(1,2))
                
    def get_reconstruction_features(self,image: torch.Tensor):
        """Returns list of features from different layers of the model.
        Args:
            image (torch.Tensor): image we want to extract features of.
        """
        reconstructed_features = []
        x = image
        
        # Loop over the slices and get different representations from different layers.
        for layers in self.slices:
            x = layers(x)
            # Unroll the image.
            reconstructed_features.append(x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))
            
        return reconstructed_features
    
    def get_style_features(self,reconstruction_features: list[torch.Tensor]):
        """Returns list of style features from different layers of the model.
        Args:
            reconstructed_features (torch.Tensor): image we want to extract features of.
        """
        style_features = []
        for feature in reconstruction_features:
            style_features.append(self.compute_gram_matrix(feature))
        return style_features
    
    def compute_L_reconstructed(self,
                   reconstruction_features_og: list[torch.Tensor],
                   reconstruction_features_up: list[torch.Tensor]):
        """Computes reconstruction loss between the features.

        Args:
            reconstruction_features_og (list[torch.Tensor]): list of reconstruction features of original picture from different layers of model.
            reconstruction_features_up (list[torch.Tensor]): list of reconstruction features of upscaled picture from different layers of model.
        """
        # Reconstruction loss with the output of the last layer in the list.
        L = torch.mean((reconstruction_features_og[-1]-reconstruction_features_up[-1])**2)
        return L
    
    def compute_L_style(self,
                        style_features_og: list[torch.Tensor],
                        style_features_up: list[torch.Tensor]):
        """Computes style loss between features. (weighted with numbers in self.weights)
        Args:
            style_features_og (list[torch.Tensor]): list of style features of original picture from different layers of model.
            style_features_up (list[torch.Tensor]): list of style features of upscaled picture from different layers of model.
        """
        L = 0.0
        for i in range(len(style_features_og)):
            L += self.weights[i]*torch.mean((style_features_og[i]-style_features_up[i])**2)
        return L
    
    def forward(self, original_image: torch.Tensor,
                upscaled_image: torch.Tensor):
        """Computes perception loss between original and upscaled image.

        Args:
            original_image (torch.Tensor): Tensor of original image.
            upscaled_image (torch.Tensor): Tensor of upscaled image.

        Returns:
            torch.Tensor: Final loss.
        """
        # Get the reconstructed features.
        reconstruction_features_up = self.get_reconstruction_features(upscaled_image)
        reconstruction_features_og = self.get_reconstruction_features(original_image)
        
        # Compute style features.
        style_features_up = self.get_style_features(reconstruction_features_up)
        style_features_og = self.get_style_features(reconstruction_features_og)
        
        # Compute the style and reconstructed losses.
        L_reconstructed = self.compute_L_reconstructed(reconstruction_features_og,
                                                       reconstruction_features_up)
        L_style = self.compute_L_style(style_features_og,
                                       style_features_up)
        
        # Calculate final loss.
        loss = self.alpha*L_reconstructed + self.beta*L_style
        
        return loss