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
        # Input x shape: (batch_size * sequence_length, channels, height, width)
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

class ImpalaModel(nn.Module):
    def __init__(self, in_channels, num_actions, feature_dim=256, lstm_hidden_size=256, height=84, width=84, device=None):
        super(ImpalaModel, self).__init__()
        self.num_actions = num_actions
        self.feature_dim = feature_dim
        self.lstm_hidden_size = lstm_hidden_size
        self.device = device
        self.cnn = ImpalaCNN(in_channels=in_channels, feature_dim=feature_dim, height=height, width=width).to(device)
        self.lstm = nn.LSTMCell(input_size=feature_dim, hidden_size=lstm_hidden_size).to(device)

        # Actor (policy) head
        self.policy_head = nn.Linear(lstm_hidden_size, num_actions).to(device)
        # Critic (value) head
        self.value_head = nn.Linear(lstm_hidden_size, 1).to(device)

    def initial_state(self, batch_size):
        # Hidden state and cell state for LSTM
        return (torch.zeros(batch_size, self.lstm_hidden_size, device=self.device),
                torch.zeros(batch_size, self.lstm_hidden_size, device=self.device))

    def forward(self, x: torch.Tensor,
                core_state: tuple[torch.Tensor, torch.Tensor]):
        """
        x          : (B, C, H, W) - one time-step for every env
        core_state : (hx, cx) where each is (B, lstm_hidden)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.cnn(x)                         # (B, feature_dim)
        hx, cx = self.lstm(x, core_state)       # LSTMCell -> (B, hidden)

        logits = self.policy_head(hx)           # (B, num_actions)
        value  = self.value_head(hx).squeeze(-1)  # (B,)

        return logits, value, (hx, cx)

# Example Usage (assuming Atari-like environment)
if __name__ == '__main__':
    # N_CHANNELS: Number of input channels (e.g., 1 for grayscale, 3 for RGB, 4 for stacked frames)
    # Typically, IMPALA uses stacked frames, e.g., 4 grayscale frames.
    # However, the in_channels to the CNN itself would be the number of channels in one observation *after* stacking.
    # For example, if you stack 4 grayscale (1 channel) frames, in_channels to CNN is 4.
    # If you stack 4 RGB (3 channels) frames, then in_channels to CNN is 12.
    # Let's assume stacked grayscale frames.
    N_CHANNELS = 4
    NUM_ACTIONS = 6  # Example for an Atari game
    IMG_HEIGHT = 84 # Example, adjust to your environment
    IMG_WIDTH = 84  # Example, adjust to your environment

    # Create model instance
    model = ImpalaModel(in_channels=N_CHANNELS, num_actions=NUM_ACTIONS, height=IMG_HEIGHT, width=IMG_WIDTH)
    print("Model created successfully.")
    print(model)

    # Dummy input
    sequence_length = 5
    batch_size = 2
    dummy_input = torch.randn(sequence_length, batch_size, N_CHANNELS, IMG_HEIGHT, IMG_WIDTH)
    initial_core_state = model.initial_state(batch_size)

    # Move model and data to device if using GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    dummy_input = dummy_input.to(device)
    initial_core_state = (initial_core_state[0].to(device), initial_core_state[1].to(device))

    # Forward pass
    try:
        logits, value, next_core_state = model(dummy_input, initial_core_state)
        print("\nForward pass successful!")
        print("Logits shape:", logits.shape)  # Expected: (sequence_length, batch_size, num_actions)
        print("Value shape:", value.shape)    # Expected: (sequence_length, batch_size, 1)
        print("Next hidden state shape:", next_core_state[0].shape) # Expected: (batch_size, lstm_hidden_size)
        print("Next cell state shape:", next_core_state[1].shape)   # Expected: (batch_size, lstm_hidden_size)

        # Check actor-learner output shapes if unbatched (sequence combined with batch)
        # T_B_logits = logits.reshape(-1, NUM_ACTIONS)
        # T_B_value = value.reshape(-1, 1)
        # print("Reshaped Logits shape (T*B, num_actions):", T_B_logits.shape)
        # print("Reshaped Value shape (T*B, 1):", T_B_value.shape)

    except ValueError as e:
        print(f"\nError during forward pass: {e}")
        print("This often happens if 'flattened_size' in ImpalaCNN is not calculated correctly "
              "for your specific input image dimensions and CNN architecture.")