import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal
from efficientnet_pytorch import EfficientNet
from torch.distributions import TransformedDistribution, TanhTransform, Independent
from torchvision import transforms

from training.lib.atari_config import Config


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer, copied from ViNT: 
    https://github.com/robodhruv/visualnav-transformer/blob/main/train/vint_train/models/vint/self_attention.py"""

    def __init__(self, d_model, max_seq_len=6):
        super().__init__()

        # Compute the positional encoding once
        pos_enc = torch.zeros(max_seq_len, d_model)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pos_enc[:, 0::2] = torch.sin(pos * div_term)
        pos_enc[:, 1::2] = torch.cos(pos * div_term)
        pos_enc = pos_enc.unsqueeze(0)

        # Register the positional encoding as a buffer to avoid it being
        # considered a parameter when saving the model
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        # Add the positional encoding to the input
        x = x + self.pos_enc[:, :x.size(1), :]
        return x
    

class MultiLayerDecoder(nn.Module):
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, dropout=0.1, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True, dropout=dropout)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        
        self.output_layers = nn.ModuleList()
        in_features = embed_dim
        for out_features in output_layers:
            self.output_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

    def forward(self, x):
        # Always apply positional encoding; avoid boolean check on nn.Module
        x = self.positional_encoding(x)
        x = self.sa_decoder(x)
        # currently, x is [batch_size, seq_len, embed_dim]
        x = x[:, -1, :] # Now x is [batch_size, embed_dim]
        for i, layer in enumerate(self.output_layers):
            x = layer(x)
            if i != len(self.output_layers) - 1: # Don't apply relu to the last layer
                x = F.relu(x)
        return x
    

class ImagePreprocessor(nn.Module):
    def __init__(self):
        super(ImagePreprocessor, self).__init__()
        self.transforms = transforms.Compose([transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def forward(self, x):
        """
        Preprocess images for EfficientNet
        Args:
            x: tensor of shape [..., H, W, C] with values in range [0, 1] (float32)
               or [..., H, W, C] with values in range [0, 255] (uint8)
        Returns:
            tensor of shape [..., C, H, W] normalized for EfficientNet
        """
        # Handle uint8 input by converting to float32 and normalizing to [0, 1]
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        
        # Convert from [..., H, W, C] to [..., C, H, W]
        x = x.permute(*range(len(x.shape) - 3), -1, -3, -2)

        # Apply ImageNet normalization
        x = self.transforms(x)
        
        return x


class ViNTActorCritic(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        # Initialize the image preprocessor
        self.image_preprocessor = ImagePreprocessor()

        # Initialize the EfficientNet model
        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b0', in_channels=config.n_channels)

        # Initialize the transformer backnbone
        self.transformer_decoder = MultiLayerDecoder(embed_dim=config.d_model, seq_len=config.frame_stack, output_layers=config.output_layers, nhead=config.n_heads, num_layers=config.n_layers, dropout=config.dropout)

        # Adapter to go from the output of the convnet to the input d_model of the transformer
        self.adapter = nn.Linear(self.efficientnet._fc.in_features, config.d_model) # 1280 to d_model

        # Initialize the actor and critic networks
        self.actor = nn.Linear(config.output_layers[-1], config.action_dim * 2) # 2 for sine and cosine
        self.critic = nn.Linear(config.output_layers[-1], 1)

    def forward(self, x):
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        batch_size, seq_len, frame_height, frame_width, n_channels = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, frame_height, frame_width, n_channels)
        
        # Preprocess images (converts to [batch_size * seq_len, n_channels, frame_height, frame_width] and normalizes)
        x = self.image_preprocessor(x)
        
        # Extract features using EfficientNet
        x = self.efficientnet.extract_features(x)
        x = self.efficientnet._avg_pooling(x) # [batch_size * seq_len, 1280, 1, 1]
        x = torch.flatten(x, start_dim=1) # [batch_size * seq_len, 1280]
        
        # Apply adapter
        x = self.adapter(x)
        
        # Reshape back to sequence format
        x = x.view(batch_size, seq_len, -1) # [batch_size, seq_len, d_model]
        
        # Apply transformer
        x = self.transformer_decoder(x)
        
        return self.actor(x), self.critic(x)
    
    def get_action_and_value(self, x):
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        action_logits, value = self.forward(x)
        
        means = action_logits[:, :self.config.action_dim]
        log_stds = action_logits[:, self.config.action_dim:]
        stds = torch.exp(log_stds.clamp(-5, 2))
        
        base_dist = Independent(Normal(means, stds), 1)
        squashed_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        
        action = squashed_dist.sample()
        log_prob = squashed_dist.log_prob(action)
        entropy = base_dist.entropy()
        
        value = value.squeeze()
        
        return action, log_prob, entropy, value
    
    def evaluate_actions(self, x, actions):
        """
        obs: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        actions: tensor of shape [batch_size, action_dim] - these are the squashed actions in [-1, 1]
        """
        logits, value = self.forward(x)

        means = logits[:, :self.config.action_dim]
        log_stds = logits[:, self.config.action_dim:]
        stds = torch.exp(log_stds.clamp(-5, 2))

        base_dist = Independent(Normal(means, stds), 1)
        squashed_dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

        # Clamp actions slightly inside the open interval (-1, 1) to avoid numerical
        # instabilities when computing the inverse tanh in log_prob. Actions that hit
        # exactly the boundaries yield +/-inf after atanh which in turn produces NaNs
        # in the subsequent computations and can corrupt the network weights.
        eps = 1e-6
        actions_clamped = actions.clamp(-1 + eps, 1 - eps)

        log_prob = squashed_dist.log_prob(actions_clamped)
        entropy = base_dist.entropy()

        return log_prob, entropy, value.squeeze(-1)