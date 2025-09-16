import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from lib.atari_config import Config
from torch.distributions import TransformedDistribution, TanhTransform, Independent, Normal, Categorical
from lib.noisy import NoisyLinear

def calculate_conv_output_size(input_size, kernel_size, stride, padding=0):
    """Calculate output size after convolution"""
    return (input_size + 2 * padding - kernel_size) // stride + 1


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer
    

class SimpleImagePreprocessor(nn.Module):
    def __init__(self):
        super(SimpleImagePreprocessor, self).__init__()

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

        batch_size, seq_len, frame_height, frame_width, n_channels = x.shape

        # Rearrange so that sequence length and channel dims are adjacent, then stack
        # Resulting shape: [batch_size, seq_len * n_channels, frame_height, frame_width]
        x = x.permute(0, 1, 4, 2, 3).reshape(batch_size, seq_len * n_channels, frame_height, frame_width)
        
        return x
    

class BaseModel(nn.Module):
    def __init__(self, config: Config):
        super(BaseModel, self).__init__()
        self.config = config

        self.image_preprocessor = SimpleImagePreprocessor()

        # Calculate the size after convolutions
        # Input: (C, H, W) = (1, 84, 84) for grayscale
        c, h, w = config.n_channels * config.frame_stack, config.output_height, config.output_width
        
        # First conv: kernel=8, stride=4
        h1 = calculate_conv_output_size(h, 8, 4)
        w1 = calculate_conv_output_size(w, 8, 4)
        
        # Second conv: kernel=4, stride=2  
        h2 = calculate_conv_output_size(h1, 4, 2)
        w2 = calculate_conv_output_size(w1, 4, 2)
        
        # Third conv: kernel=3, stride=1
        h3 = calculate_conv_output_size(h2, 3, 1)
        w3 = calculate_conv_output_size(w2, 3, 1)
        
        # Final feature size: 64 channels * h3 * w3
        conv_output_size = 64 * h3 * w3

        self.feature_net = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(conv_output_size, self.config.d_model)),
            nn.ReLU(),
            layer_init(nn.Linear(self.config.d_model, self.config.d_model))
        )

    def forward(self, x: torch.Tensor, preprocess: bool = True) -> torch.Tensor:
        if preprocess:
            x = self.image_preprocessor(x)
        x = self.feature_net(x)
        return x

class SimpleModel(nn.Module):
    def __init__(self, config: Config):
        super(SimpleModel, self).__init__()
        self.config = config

        self.base_model = BaseModel(config)

        LinearCls = NoisyLinear if config.use_noisy_linear else nn.Linear

        if self.config.continuous_action:
            # Continuous policy: mean vector and state-independent log std
            self.actor_loc = LinearCls(self.config.d_model, self.config.action_dim)
            self.actor_logstd = nn.Parameter(
                torch.ones(1, self.config.action_dim) * config.log_std_start
            )
        else:
            # Discrete policy: logits over actions
            self.actor_logits = LinearCls(self.config.d_model, self.config.action_dim)

        
        if self.config.use_dual_value_heads and self.config.rnd_enabled:
            self.critic_ext = LinearCls(self.config.d_model, 1)  # Extrinsic value head
            self.critic_int = LinearCls(self.config.d_model, 1)  # Intrinsic value head
        else:
            self.critic = LinearCls(self.config.d_model, 1)  # Unbounded value output
    
    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """
        x = self.base_model(x)
        if self.config.use_dual_value_heads and self.config.rnd_enabled:
            return self.critic_ext(x), self.critic_int(x)
        else:
            return self.critic(x)
        
    def get_value_ext(self, x: torch.Tensor) -> torch.Tensor:
        """Get extrinsic value estimate"""
        if not self.config.use_dual_value_heads:
            return self.get_value(x)
        x = self.base_model(x)
        return self.critic_ext(x)

    def get_value_int(self, x: torch.Tensor) -> torch.Tensor:
        """Get intrinsic value estimate"""
        if not self.config.use_dual_value_heads:
            return torch.zeros_like(self.get_value(x))
        x = self.base_model(x)
        return self.critic_int(x)
    
    def get_action_and_value(self, x: torch.Tensor, action: torch.Tensor = None) -> torch.Tensor:
        """
        x: tensor of shape [batch_size, seq_len, frame_height, frame_width, n_channels]
        """

        x = self.base_model(x)

        if self.config.continuous_action:
            # Continuous action policy: Tanh-Normal
            action_loc = self.actor_loc(x)
            action_logstd = self.actor_logstd.expand_as(action_loc)
            action_logstd = torch.clamp(
                action_logstd,
                min=self.config.log_std_min,
                max=self.config.log_std_max,
            )
            action_std = torch.exp(action_logstd)
            base_dist = Independent(Normal(action_loc, action_std), 1)
            dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])

            if action is None:
                action = dist.sample()
            else:
                eps = 1e-6
                action = action.clamp(-1 + eps, 1 - eps)

            logprob = dist.log_prob(action)
            entropy = base_dist.entropy()
        else:
            # Discrete action policy: Categorical over logits
            logits = self.actor_logits(x)
            dist = Categorical(logits=logits)

            if action is None:
                action = dist.sample()
            else:
                action = action.long()

            # log_prob expects shape (batch,)
            logprob = dist.log_prob(action)
            entropy = dist.entropy()

        if self.config.use_dual_value_heads and self.config.rnd_enabled:
            value_ext = self.critic_ext(x)
            value_int = self.critic_int(x)
            value = value_ext + value_int
            return action, logprob, entropy, value, value_ext, value_int
        else:
            value = self.critic(x)
            return action, logprob, entropy, value
        
    def reset_noise(self):
        """Reset the noise of the actor and critic networks"""
        if not self.config.use_noisy_linear:
            return
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

class RNDNetwork(nn.Module):
    def __init__(self, config: Config):
        super(RNDNetwork, self).__init__()
        self.config = config
        self.feature_net = BaseModel(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_net(x, preprocess=False)
    

class RNDTargetNetwork(RNDNetwork):
    def __init__(self, config: Config):
        super(RNDTargetNetwork, self).__init__(config)
        # Freeze the target network
        for param in self.parameters():
            param.requires_grad = False


class RNDPredictorNetwork(RNDNetwork):
    def __init__(self, config: Config):
        super(RNDPredictorNetwork, self).__init__(config)


class RNDModule(nn.Module):
    def __init__(self, config: Config):
        super(RNDModule, self).__init__()
        self.config = config

        self.predictor_net = RNDPredictorNetwork(config)
        self.target_net = RNDTargetNetwork(config)

        self.preproc = self.predictor_net.feature_net.image_preprocessor  

        # Running statistics for observation normalization
        self.register_buffer('obs_running_mean', torch.zeros(1))
        self.register_buffer('obs_running_var', torch.ones(1))
        self.register_buffer('obs_count', torch.zeros(1))
        
        # Running statistics for reward normalization
        self.register_buffer('reward_running_mean', torch.zeros(1))
        self.register_buffer('reward_running_var', torch.ones(1))
        self.register_buffer('reward_count', torch.zeros(1))

    def normalize_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Normalize the observation using running mean and variance
        """
        if self.training:
            # Update running statistics
            batch_mean = obs.mean()
            batch_var = obs.var()
            batch_count = obs.numel()
            
            delta = batch_mean - self.obs_running_mean
            total_count = self.obs_count + batch_count
            
            new_mean = self.obs_running_mean + delta * batch_count / total_count
            m_a = self.obs_running_var * self.obs_count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta.pow(2) * self.obs_count * batch_count / total_count
            new_var = M2 / total_count
            
            self.obs_running_mean.copy_(new_mean)
            self.obs_running_var.copy_(new_var)
            self.obs_count.copy_(total_count)
        
        # Normalize and clip
        normalized = (obs - self.obs_running_mean) / torch.sqrt(self.obs_running_var + 1e-8)
        return torch.clamp(normalized, -5, 5)
    
    def compute_intrinsic_reward(self, x):
        """Compute intrinsic reward from prediction error"""
        x = self.preproc(x)
        x = self.normalize_obs(x)
        
        # Get features from both networks
        with torch.no_grad():
            target_features = self.target_net(x)
        
        predicted_features = self.predictor_net(x)
        
        # Compute prediction error (intrinsic reward)
        intrinsic_reward = 0.5 * (predicted_features - target_features).pow(2).sum(dim=1)
        
        return intrinsic_reward
    
    def normalize_reward(self, reward):
        """Normalize intrinsic rewards using running statistics"""
        if self.training and reward.numel() > 0:
            # Update running statistics
            batch_mean = reward.mean()
            batch_var = reward.var()
            batch_count = reward.numel()
            
            delta = batch_mean - self.reward_running_mean
            total_count = self.reward_count + batch_count
            
            if total_count > 0:
                new_mean = self.reward_running_mean + delta * batch_count / total_count
                m_a = self.reward_running_var * self.reward_count
                m_b = batch_var * batch_count
                M2 = m_a + m_b + delta.pow(2) * self.reward_count * batch_count / total_count
                new_var = M2 / total_count
                
                self.reward_running_mean.copy_(new_mean)
                self.reward_running_var.copy_(new_var)
                self.reward_count.copy_(total_count)
        
        # Normalize reward
        return reward / torch.sqrt(self.reward_running_var + 1e-8)

    def forward(self, obs):
        """Forward pass returning normalized intrinsic reward"""
        intrinsic_reward = self.compute_intrinsic_reward(obs)
        normalized_reward = self.normalize_reward(intrinsic_reward)
        return normalized_reward