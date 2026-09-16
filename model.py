"""
RoshamboNet: PyTorch implementation of the 64x64 Tiny CNN for Rock-Paper-Scissors.
Faithfully ported from the original dextra-roshambo-python architecture:
- 64x64 single-channel input
- Conv2D(16, 5x5) + AvgPool2D(2, 2) -> (16, 30, 30)
- 3x [Conv2D(3x3) + AvgPool2D(2, 2)] with channels doubling (32, 64, 128)
- Conv2D(128, 1x1) + AvgPool2D(2, 2) -> (128, 1, 1)
- Flatten + Linear(128, 4) -> 4 classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Class label mappings matching the dataset expectation
CLASS_NAMES = ["0_rock", "1_paper", "2_scissors", "3_background"]
LABEL_TO_SYMBOL = {
    0: "rock",
    1: "paper",
    2: "scissors",
    3: "background",
}
SYMBOL_TO_LABEL = {v: k for k, v in LABEL_TO_SYMBOL.items()}

# Invincible Counter-Move logic:
# User Rock (0)     -> AI Paper (1)
# User Paper (1)    -> AI Scissors (2)
# User Scissors (2) -> AI Rock (0)
# User Background (3)-> AI None / Idle
COUNTER_MOVES = {
    0: {"ai_label": 1, "ai_symbol": "paper", "desc": "Paper covers Rock", "icon": "✋"},
    1: {"ai_label": 2, "ai_symbol": "scissors", "desc": "Scissors cuts Paper", "icon": "✌️"},
    2: {"ai_label": 0, "ai_symbol": "rock", "desc": "Rock crushes Scissors", "icon": "✊"},
    3: {"ai_label": 3, "ai_symbol": "background", "desc": "Waiting for gesture...", "icon": "⏳"},
}


class RoshamboNet(nn.Module):
    """
    Ultra-lightweight CNN (<150k parameters) designed for sub-3ms CPU inference
    on 64x64 binary motion event frames.
    """
    def __init__(self, num_classes=4, pooling="avg"):
        super(RoshamboNet, self).__init__()
        
        PoolLayer = nn.AvgPool2d if pooling == "avg" else nn.MaxPool2d
        
        # Block 1: 64x64 -> 60x60 -> 30x30
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=0)
        self.pool1 = PoolLayer(kernel_size=2, stride=2)
        
        # Block 2: 30x30 -> 28x28 -> 14x14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.pool2 = PoolLayer(kernel_size=2, stride=2)
        
        # Block 3: 14x14 -> 12x12 -> 6x6
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.pool3 = PoolLayer(kernel_size=2, stride=2)
        
        # Block 4: 6x6 -> 4x4 -> 2x2
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.pool4 = PoolLayer(kernel_size=2, stride=2)
        
        # Block 5: 2x2 -> 2x2 -> 1x1
        self.conv5 = nn.Conv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.pool5 = PoolLayer(kernel_size=2, stride=2)
        
        # Fully Connected
        self.fc = nn.Linear(128 * 1 * 1, num_classes)
        
    def forward(self, x):
        # x shape: (B, 1, 64, 64)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class MajorityVote:
    """
    Exact port of majority_vote temporal filter from dextra-roshambo-python consumer.py.
    Provides temporal stabilization over a sliding window (default 5 frames).
    """
    def __init__(self, window_length=5, num_classes=4):
        self.window_length = window_length
        self.num_classes = num_classes
        self.ptr = 0
        self.cirbuf = [-1] * self.window_length
        self.cmdcnts = [0] * self.num_classes
        self.num_predictions = 0

    def new_prediction_and_vote(self, symbol: int):
        if 0 <= symbol < self.num_classes:
            self.num_predictions += 1
            idx = self.ptr
            if self.num_predictions > self.window_length:
                old_symbol = self.cirbuf[idx]
                if old_symbol >= 0:
                    self.cmdcnts[old_symbol] -= 1
            self.cirbuf[idx] = symbol
            self.cmdcnts[symbol] += 1
            self.ptr = (self.ptr + 1) % self.window_length

        return self.vote()

    def vote(self):
        majority_count = self.window_length // 2 + 1  # 3 for window_length=5
        max_idx = int(torch.tensor(self.cmdcnts).argmax().item())
        if self.cmdcnts[max_idx] >= majority_count:
            return max_idx
        return None
