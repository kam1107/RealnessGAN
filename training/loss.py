import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import sys
import sys
import datetime
import time

def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

class CategoricalLoss(nn.Module):
    def __init__(self, atoms=51, v_max=10, v_min=-10):
        super(CategoricalLoss, self).__init__()

        self.atoms = atoms
        self.v_max = v_max
        self.v_min = v_min
        self.supports = torch.linspace(v_min, v_max, atoms).view(1, 1, atoms) # RL: [bs, #action, #quantiles]
        self.delta = (v_max - v_min) / (atoms - 1)

    def to(self, device):
        self.device = device
        self.supports = self.supports.to(device)

    def forward(self, anchor, feature, skewness=0.0):
        batch_size = feature.shape[0]
        skew = torch.zeros((batch_size, self.atoms)).to(self.device).fill_(skewness)

        # experiment to adjust KL divergence between positive/negative anchors
        Tz = skew + self.supports.view(1, -1) * torch.ones((batch_size, 1)).to(torch.float).view(-1, 1).to(self.device)
        Tz = Tz.clamp(self.v_min, self.v_max)
        b = (Tz - self.v_min) / self.delta
        l = b.floor().to(torch.int64)
        u = b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.atoms - 1)) * (l == u)] += 1
        offset = torch.linspace(0, (batch_size - 1) * self.atoms, batch_size).to(torch.int64).unsqueeze(dim=1).expand(batch_size, self.atoms).to(self.device)
        skewed_anchor = torch.zeros(batch_size, self.atoms).to(self.device)
        skewed_anchor.view(-1).index_add_(0, (l + offset).view(-1), (anchor * (u.float() - b)).view(-1))  
        skewed_anchor.view(-1).index_add_(0, (u + offset).view(-1), (anchor * (b - l.float())).view(-1))  

        loss = -(skewed_anchor * (feature + 1e-16).log()).sum(-1).mean()

        return loss

