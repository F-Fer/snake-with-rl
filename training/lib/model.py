import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResidualBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        inputs = x
        x = F.relu(x)
        x = self.conv0(x)
        x = F.relu(x)
        x = self.conv1(x)
        return x + inputs

class ImpalaCNN(nn.Module):
    def __init__(self, in_channels, feature_dim=256, height=84, width=84):
        super(ImpalaCNN, self).__init__()
        self.feature_dim = feature_dim
        self.blocks = []
        self.channels_configs = [16, 32, 32] # Number of channels for each block stack

        current_channels = in_channels
        for i, num_channels in enumerate(self.channels_configs):
            layers = []
            layers.append(nn.Conv2d(in_channels=current_channels, out_channels=num_channels, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            layers.append(ResidualBlock(num_channels))
            layers.append(ResidualBlock(num_channels))
            self.blocks.append(nn.Sequential(*layers))
            current_channels = num_channels

        self.final_convs = nn.ModuleList(self.blocks)
        self.flattened_size = self._calculate_flattened_size(height, width)

        self.fc = nn.Linear(self.flattened_size, feature_dim)

    def forward(self, x):
        # Input x shape: (batch_size, channels, height, width)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
            
        for block_stack in self.final_convs:
            x = block_stack(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1) # Flatten
        if x.size(1) != self.flattened_size:
            raise ValueError(f"Expected flattened size {self.flattened_size}, but got {x.size(1)}. "
                             f"Please adjust 'flattened_size' in ImpalaCNN based on your input dimensions "
                             f"and CNN architecture. Input shape to CNN was {list(x.shape)} before flatten.")
        x = self.fc(x)
        x = F.relu(x)
        return x
    
    def _calculate_flattened_size(self, height, width):
        """
        Calculate the flattened size of the output of the CNN.
        """
        for _ in range(len(self.channels_configs)):
            height = (height - 3 + 2*1) // 2 + 1
            width = (width - 3 + 2*1) // 2 + 1
        return self.channels_configs[-1] * height * width


class PolicyValueHead(nn.Module):
    """Separate actor and critic heads for the LSTM output"""
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.policy_head = nn.Linear(input_size, num_actions)
        self.value_head = nn.Linear(input_size, 1)
    
    def forward(self, x):
        logits = self.policy_head(x)
        value = self.value_head(x).squeeze(-1)
        
        # Split logits into loc and scale for TanhNormal distribution
        loc, scale = torch.chunk(logits, 2, dim=-1)
        scale = F.softplus(scale) + 1e-4  # Ensure positive scale
        
        return {
            "loc": loc,
            "scale": scale,
            "state_value": value
        }


# Example Usage (if running this file directly)
if __name__ == '__main__':
    # Test the CNN backbone
    N_CHANNELS = 6  # Example for stacked frames
    IMG_HEIGHT = 84
    IMG_WIDTH = 84
    FEATURE_DIM = 256

    # Create CNN backbone
    cnn = ImpalaCNN(in_channels=N_CHANNELS, feature_dim=FEATURE_DIM, height=IMG_HEIGHT, width=IMG_WIDTH)
    print("CNN backbone created successfully.")
    print(cnn)

    # Test with dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, N_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    
    try:
        cnn_output = cnn(dummy_input)
        print(f"\nCNN forward pass successful!")
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {cnn_output.shape}")
        print(f"Expected output shape: ({batch_size}, {FEATURE_DIM})")
        
        # Test policy/value head
        policy_value_head = PolicyValueHead(FEATURE_DIM, num_actions=4)  # 4 = 2 actions * 2 (loc + scale)
        head_output = policy_value_head(cnn_output)
        print(f"\nPolicy/Value head output:")
        print(f"loc shape: {head_output['loc'].shape}")
        print(f"scale shape: {head_output['scale'].shape}")
        print(f"state_value shape: {head_output['state_value'].shape}")
        
    except ValueError as e:
        print(f"\nError during forward pass: {e}")