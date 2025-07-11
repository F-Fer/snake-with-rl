import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.distributions import Normal
from efficientnet_pytorch import EfficientNet

from training.ppo_transformer.lib.config import Config


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
    def __init__(self, embed_dim=512, seq_len=6, output_layers=[256, 128, 64], nhead=8, num_layers=8, ff_dim_factor=4):
        super(MultiLayerDecoder, self).__init__()
        self.positional_encoding = PositionalEncoding(embed_dim, max_seq_len=seq_len)
        self.sa_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=nhead, dim_feedforward=ff_dim_factor*embed_dim, activation="gelu", batch_first=True, norm_first=True)
        self.sa_decoder = nn.TransformerEncoder(self.sa_layer, num_layers=num_layers)
        
        self.output_layers = nn.ModuleList()
        in_features = embed_dim
        for out_features in output_layers:
            self.output_layers.append(nn.Linear(in_features, out_features))
            in_features = out_features

    def forward(self, x):
        if self.positional_encoding: x = self.positional_encoding(x)
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
        # EfficientNet ImageNet normalization parameters
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

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
        x = (x - self.mean) / self.std
        
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
        self.transformer_decoder = MultiLayerDecoder(embed_dim=config.d_model, seq_len=config.frame_stack, output_layers=config.output_layers)

        # Adapter to go from the output of the convnet to the input d_model of the transformer
        self.adapter = nn.Linear(1280, config.d_model) # TODO: change this to the output of the convnet dynamically or smt

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
        action_logits, value = self.forward(x) # [batch_size, 4]
        
        # Split the action logits into mean and log_std
        # We interpret the spread on log scale, so we can exponentiate it and therefore guarantee positive values
        cos_mean, sin_mean, cos_log_std, sin_log_std = torch.chunk(action_logits, 4, dim=-1)
        cos_std = torch.exp(cos_log_std.clamp(-5, 2))
        sin_std = torch.exp(sin_log_std.clamp(-5, 2))
        
        # Create action distributions
        cos_action_distribution = torch.distributions.Normal(cos_mean, cos_std)
        sin_action_distribution = torch.distributions.Normal(sin_mean, sin_std)

        # Sample from the action distributions (raw actions before squashing)
        cos_action_raw = cos_action_distribution.sample()
        sin_action_raw = sin_action_distribution.sample()

        # Apply tanh squashing to ensure actions are in [-1, 1]
        cos_action = torch.tanh(cos_action_raw)
        sin_action = torch.tanh(sin_action_raw)

        # Combine the squashed actions
        action = torch.cat([cos_action, sin_action], dim=-1)

        # Calculate log probabilities with tanh correction
        # log_prob = log_prob_raw - log(1 - tanh^2(raw_action))
        sin_log_prob_raw = sin_action_distribution.log_prob(sin_action_raw).sum(dim=-1)
        cos_log_prob_raw = cos_action_distribution.log_prob(cos_action_raw).sum(dim=-1)
        
        # Tanh correction term
        sin_tanh_correction = torch.log(1 - torch.tanh(sin_action_raw)**2 + 1e-6).sum(dim=-1)
        cos_tanh_correction = torch.log(1 - torch.tanh(cos_action_raw)**2 + 1e-6).sum(dim=-1)
        
        sin_log_prob = sin_log_prob_raw - sin_tanh_correction
        cos_log_prob = cos_log_prob_raw - cos_tanh_correction
        total_log_prob = sin_log_prob + cos_log_prob

        # Calculate entropy (note: entropy changes due to tanh transformation)
        sin_entropy = sin_action_distribution.entropy().sum(dim=-1)
        cos_entropy = cos_action_distribution.entropy().sum(dim=-1)
        entropy = sin_entropy + cos_entropy

        # Calculate value
        value = value.squeeze()
            
        return action, total_log_prob, entropy, value
    
    def evaluate_actions(self, obs, actions):
       """
       obs: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
       actions: tensor of shape [batch_size, action_dim] - these are the squashed actions in [-1, 1]
       """
       logits, value = self.forward(obs)
       cos_mean, sin_mean, cos_log_std, sin_log_std = torch.chunk(logits, 4, dim=-1)
       cos_std = torch.exp(cos_log_std.clamp(-5, 2))
       sin_std = torch.exp(sin_log_std.clamp(-5, 2))
       
       # Create distributions for raw (unsquashed) actions
       cos_dist = Normal(cos_mean, cos_std)
       sin_dist = Normal(sin_mean, sin_std)
       
       # Convert squashed actions back to raw actions using inverse tanh
       cos_action_squashed = actions[:, 0:1]
       sin_action_squashed = actions[:, 1:2]
       
       # Clamp to avoid numerical issues with atanh
       cos_action_squashed = torch.clamp(cos_action_squashed, -0.999999, 0.999999)
       sin_action_squashed = torch.clamp(sin_action_squashed, -0.999999, 0.999999)
       
       cos_action_raw = torch.atanh(cos_action_squashed)
       sin_action_raw = torch.atanh(sin_action_squashed)
       
       # Calculate log probabilities for raw actions
       cos_log_prob_raw = cos_dist.log_prob(cos_action_raw).sum(-1)
       sin_log_prob_raw = sin_dist.log_prob(sin_action_raw).sum(-1)
       
       # Apply tanh correction
       cos_tanh_correction = torch.log(1 - cos_action_squashed**2 + 1e-6).sum(-1)
       sin_tanh_correction = torch.log(1 - sin_action_squashed**2 + 1e-6).sum(-1)
       
       cos_log_prob = cos_log_prob_raw - cos_tanh_correction
       sin_log_prob = sin_log_prob_raw - sin_tanh_correction
       log_prob = cos_log_prob + sin_log_prob
       
       # Calculate entropy
       cos_entropy = cos_dist.entropy().sum(-1)
       sin_entropy = sin_dist.entropy().sum(-1)
       entropy = cos_entropy + sin_entropy
       
       return log_prob, entropy, value.squeeze(-1)