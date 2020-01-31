import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
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

def learnD_Realness(param, D, G, optimizerD, random_sample, Triplet_Loss, x, anchor1, anchor0):
    device = 'cuda' if param.cuda else 'cpu'
    z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
    z = z.to(device)

    for p in D.parameters():
        p.requires_grad = True

    for t in range(param.D_updates):
        D.zero_grad()
        optimizerD.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            images, _ = random_sample.__next__()
            num_outcomes = Triplet_Loss.atoms
            x.copy_(images)
            del images

            anchor_real = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor1, dtype=torch.float).to(device)
            anchor_fake = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor0, dtype=torch.float).to(device)

            # real images
            feat_real = D(x).log_softmax(1).exp()

            # fake images
            z.normal_(0, 1)
            imgs_fake = G(z)
            feat_fake = D(imgs_fake.detach()).log_softmax(1).exp()

            lossD_real = Triplet_Loss(anchor_real, feat_real, skewness=param.positive_skew)
            lossD_real.backward()

            lossD_fake = Triplet_Loss(anchor_fake, feat_fake, skewness=param.negative_skew)
            lossD_fake.backward()

            lossD = lossD_real + lossD_fake

        optimizerD.step()

    return lossD



