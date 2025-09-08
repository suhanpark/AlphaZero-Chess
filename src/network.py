import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(ch)
        self.c2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(ch)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(x + y)

class PolicyValueNet(nn.Module):
    def __init__(self, in_channels=18, width=128, blocks=10, action_size=4096):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU()
        )
        self.blocks = nn.Sequential(*[ResBlock(width) for _ in range(blocks)])

        # policy head
        self.p_conv = nn.Conv2d(width, 32, 1, bias=False)
        self.p_bn   = nn.BatchNorm2d(32)
        self.p_fc   = nn.Linear(32*8*8, action_size)

        # value head
        self.v_conv = nn.Conv2d(width, 32, 1, bias=False)
        self.v_bn   = nn.BatchNorm2d(32)
        self.v_fc1  = nn.Linear(32*8*8, 128)
        self.v_fc2  = nn.Linear(128, 1)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        p = F.relu(self.p_bn(self.p_conv(x)))
        p = p.view(p.size(0), -1)
        p = self.p_fc(p)  # logits

        v = F.relu(self.v_bn(self.v_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.v_fc1(v))
        v = torch.tanh(self.v_fc2(v))
        return p, v.squeeze(-1)
