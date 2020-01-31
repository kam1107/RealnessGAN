import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transf
import torchvision.models as models
import torchvision.utils as vutils
import torch.nn.utils.spectral_norm as spectral_norm

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

class GAN_G(nn.Module):
    def __init__(self, param):
        super(GAN_G, self).__init__()
        model = []

        model.append(torch.nn.Linear(param.z_size, param.G_h_size))
        model.append(torch.nn.BatchNorm1d(param.G_h_size))
        model.append(torch.nn.ReLU())

        model.append(torch.nn.Linear(param.G_h_size, param.G_h_size))
        model.append(torch.nn.BatchNorm1d(param.G_h_size))
        model.append(torch.nn.ReLU())

        model.append(torch.nn.Linear(param.G_h_size, param.G_h_size))
        model.append(torch.nn.BatchNorm1d(param.G_h_size))
        model.append(torch.nn.ReLU())

        model.append(torch.nn.Linear(param.G_h_size, param.G_h_size))
        model.append(torch.nn.BatchNorm1d(param.G_h_size))
        model.append(torch.nn.ReLU())

        model.append(torch.nn.Linear(param.G_h_size, 2))

        model = torch.nn.Sequential(*model)
        self.model = model
        self.param = param
    
    def forward(self, input):
        input = input.squeeze()
        output = self.model(input)
        return output


class Maxout(nn.Module):
    def __init__(self, d_in, d_out, pool_size):
        super().__init__()
        self.d_in, self.d_out, self.pool_size = d_in, d_out, pool_size
        self.lin = nn.Linear(d_in, d_out * pool_size)

    def forward(self, inputs):
        shape = list(inputs.size())
        shape[-1] = self.d_out
        shape.append(self.pool_size)
        max_dim = len(shape) - 1
        out = self.lin(inputs)
        m, i = out.view(*shape).max(max_dim)
        return m

class GAN_D(nn.Module):
    def __init__(self, param):
        super(GAN_D, self).__init__()
        model = []

        model.append(Maxout(2, param.D_h_size, 5))
        model.append(Maxout(param.D_h_size, param.D_h_size, 5))
        model.append(Maxout(param.D_h_size, param.D_h_size, 5))
        model.append(nn.Linear(param.D_h_size, param.num_outcomes))

        model = torch.nn.Sequential(*model)
        self.model = model
        self.param = param
    
    def forward(self, input):
        out = self.model(input).view(-1, self.param.num_outcomes)
        return out


