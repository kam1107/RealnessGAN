import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
import math
import sys
import datetime
import time

from collections import namedtuple

def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

def learnG_Realness(param, D, G, optimizerG, random_sample, Triplet_Loss, x, anchor1):
    device = 'cuda' if param.cuda else 'cpu'
    z = torch.FloatTensor(param.batch_size, param.z_size, 1, 1)
    z = z.to(device)

    G.train()
    for p in D.parameters():
        p.requires_grad = False

    for t in range(param.G_updates):
        G.zero_grad()
        optimizerG.zero_grad()

        # gradients are accumulated through subiters
        for _ in range(param.effective_batch_size // param.batch_size):
            images, _ = random_sample.__next__()
            x.copy_(images)
            del images

            num_outcomes = Triplet_Loss.atoms
            anchor_real = torch.zeros((x.shape[0], num_outcomes), dtype=torch.float).to(device) + torch.tensor(anchor1, dtype=torch.float).to(device)

            # real images
            feat_real = D(x).log_softmax(1).exp()

            # fake images
            z.normal_(0, 1)
            imgs_fake = G(z)
            feat_fake = D(imgs_fake).log_softmax(1).exp()

            # compute loss
            if param.relativisticG:
                lossG = Triplet_Loss(feat_real, feat_fake)
            else:
                lossG = Triplet_Loss(anchor_real, feat_fake, skewness=param.positive_skew)
            lossG.backward()

        optimizerG.step()
    
    return lossG


            



        


            
                

            

