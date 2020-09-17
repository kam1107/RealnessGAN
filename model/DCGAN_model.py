import torch
import torch.nn as nn
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

class DCGAN_G(nn.Module):
    def __init__(self, param):
        super(DCGAN_G, self).__init__()
        model = []

        self.param = param
        mult = self.param.image_size // 8

        # start block
        model.append(nn.ConvTranspose2d(self.param.z_size, self.param.G_h_size * mult, kernel_size=4, stride=1, padding=0, bias=False))
        model.append(nn.BatchNorm2d(self.param.G_h_size * mult))

        model.append(nn.ReLU())

        # middel block
        while mult > 1:
            model.append(nn.ConvTranspose2d(self.param.G_h_size * mult, self.param.G_h_size * (mult//2), kernel_size=4, stride=2, padding=1, bias=False))
            model.append(nn.BatchNorm2d(self.param.G_h_size * (mult//2)))

            model.append(nn.ReLU())

            mult = mult // 2

        # end block
        model.append(nn.ConvTranspose2d(self.param.G_h_size, self.param.n_channels, kernel_size=4, stride=2, padding=1, bias=False))
        model.append(nn.Tanh())

        self.model = nn.Sequential(*model)

    def forward(self, input):
        output = self.model(input)
        return output


class DCGAN_D(nn.Module):
    def __init__(self, param):
        super(DCGAN_D, self).__init__()
        self.param = param
        model = []

        # start block
        model.append(spectral_norm(nn.Conv2d(self.param.n_channels, self.param.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        image_size_new = self.param.image_size // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(spectral_norm(nn.Conv2d(self.param.D_h_size * mult, self.param.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.2, inplace=True))

            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(param.D_h_size * mult * 4 * 4)
        out_size = self.param.num_outcomes 
        self.fc = spectral_norm(nn.Linear(in_size, out_size, bias=False))

        # resampling trick
        self.reparam = spectral_norm(nn.Linear(in_size, out_size * 2, bias=False))

    def forward(self, input):
        y = self.model(input)

        y = y.view(-1, self.param.D_h_size * self.mult * 4 * 4)
        output = self.fc(y).view(-1, self.param.num_outcomes)

        # re-parameterization trick
        if self.param.use_adaptive_reparam:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(input.shape[0], self.param.num_outcomes, 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.param.num_outcomes)

        return output

