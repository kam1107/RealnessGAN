import os
import numpy as np
import random
import math
import gc
import pickle
import time
import sys
import datetime

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
from torch.utils.data import DataLoader as DataLoader

from training.learnD import learnD_Realness
from training.learnG import learnG_Realness
from training.loss import CategoricalLoss


def print_now(cmd, file=None):
	time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	if file is None:
		print('%s %s' % (time_now, cmd))
	else:
		print_str = '%s %s' % (time_now, cmd)
		print(print_str, file=file)
	sys.stdout.flush()

from options import get_args
param = get_args()

start = time.time()

if param.load_ckpt is None:
    if param.gen_extra_images > 0 and not os.path.exists(f"{param.extra_folder}"):
        os.mkdir(f"{param.extra_folder}")
    print_now(param)

if param.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    device = 'cuda'

random.seed(param.seed)
np.random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
    torch.cuda.manual_seed_all(param.seed)

# import dataset (mixture of Gaussians)
import pickle
with open(param.input_folder, 'rb') as f:
    data = pickle.load(f)

import random
class DataProvider:
    def __init__(self, data, batch_size):
        self.batch_size = batch_size
        self.data = data
        self.build()
    def build(self):
        random.shuffle(self.data)
        self.iter = iter(self.data)
    def __next__(self):
        try:
            batch = []
            for i in range(self.batch_size):
                batch.append(self.iter.__next__())
            batch = torch.tensor(batch)
            return batch, None
        except StopIteration:
            self.build()
            batch = []
            for i in range(self.batch_size):
                batch.append(self.iter.__next__())
            batch = torch.tensor(batch)
            return batch, None

random_sample = DataProvider(data, param.batch_size)

from model.GAN_model import GAN_G, GAN_D
G = GAN_G(param)
D = GAN_D(param)
print_now('Using feature size of {}'.format(param.num_outcomes))
Triplet_Loss = CategoricalLoss(atoms=param.num_outcomes, v_max=param.positive_skew, v_min=param.negative_skew)

if param.n_gpu > 1:
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

print_now("Initialized weights")
G.apply(weights_init)
D.apply(weights_init)

# to cuda
G = G.to(device)
D = D.to(device)
Triplet_Loss.to(device)
x = torch.FloatTensor(param.batch_size, 2).to(device)
optimizerD = torch.optim.Adam(D.parameters(), lr=param.lr_D, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay, eps=param.adam_eps)
optimizerG = torch.optim.Adam(G.parameters(), lr=param.lr_G, betas=(param.beta1, param.beta2), weight_decay=param.weight_decay)
decayD = torch.optim.lr_scheduler.ExponentialLR(optimizerD, gamma=1-param.decay)
decayG = torch.optim.lr_scheduler.ExponentialLR(optimizerG, gamma=1-param.decay)

if param.load_ckpt:
    checkpoint = torch.load(param.load_ckpt)
    current_set_images = checkpoint['current_set_images']
    iter_offset = checkpoint['i']
    G.load_state_dict(checkpoint['G_state'])
    D.load_state_dict(checkpoint['D_state'], strict=False)
    optimizerG.load_state_dict(checkpoint['G_optimizer'])
    optimizerD.load_state_dict(checkpoint['D_optimizer'])
    decayG.load_state_dict(checkpoint['G_scheduler'])
    decayD.load_state_dict(checkpoint['D_scheduler'])
    del checkpoint
    print_now(f'Resumed from iteration {current_set_images * param.gen_every}.')
else:
    current_set_images = 0
    iter_offset = 0

print_now(G)
print_now(D)

# define anchors
gauss = np.random.normal(0, 0.1, 1000)
count, bins = np.histogram(gauss, param.num_outcomes)
anchor0 = count / sum(count)

unif = np.random.uniform(-1, 1, 1000)
count, bins = np.histogram(unif, param.num_outcomes)
anchor1 = count / sum(count)

for i in range(iter_offset, param.total_iters):
    print('***** start training iter %d *******'%i)
    D.train()
    G.train()

    lossD = learnD_Realness(param, D, G, optimizerD, random_sample, Triplet_Loss, x, anchor1, anchor0)
    lossG = learnG_Realness(param, D, G, optimizerG, random_sample, Triplet_Loss, x, anchor1)

    decayD.step()
    decayG.step()

    if i < 1000 or (i+1) % 100 == 0:
        end = time.time()
        fmt = '[%d / %d] SD: %d Diff: %.4f loss_D: %.4f loss_G: %.4f time:%.2f'
        s = fmt % (i+1, param.total_iters, param.seed,
                    -lossD.data.item() + lossG.data.item() if (lossD is not None) and (lossG is not None) else -1.0,
                    lossD.data.item()                      if lossD is not None else -1.0,
                    lossG.data.item()                      if lossG is not None else -1.0,
                    end - start)
        print_now(s)

    if (i+1) % param.gen_every == 0:
        current_set_images += 1
        if not os.path.exists('%s/models/' % (param.extra_folder)):
            os.mkdir('%s/models/' % (param.extra_folder))
        torch.save({
            'i': i + 1,
            'current_set_images': current_set_images,
            'G_state': G.state_dict(),
            'D_state': D.state_dict(),
            'G_optimizer': optimizerG.state_dict(),
            'D_optimizer': optimizerD.state_dict(),
            'G_scheduler': decayG.state_dict(),
            'D_scheduler': decayD.state_dict(),
        }, '%s/models/state_%02d.pth' % (param.extra_folder, current_set_images))
        print_now('Model saved.')

        if os.path.exists('%s/%01d/' % (param.extra_folder, current_set_images)):
            for root, dirs, files in os.walk('%s/%01d/' % (param.extra_folder, current_set_images)):
                for f in files:
                    os.unlink(os.path.join(root, f))
        else:
            os.mkdir('%s/%01d/' % (param.extra_folder, current_set_images))

        G.eval()
        extra_batch = 100
        with torch.no_grad():
            ext_curr = 0
            z_extra = torch.FloatTensor(extra_batch, param.z_size, 1, 1)
            z_extra = z_extra.to(device)
            fake_test_list = []
            for ext in range(int(param.gen_extra_images/extra_batch)):
                fake_test = G(z_extra.normal_(0, 1)).squeeze()
                fake_test = fake_test.cpu().clone().numpy()
                fake_test_list.extend(fake_test)
            with open('%s/%01d/extra.pk' % (param.extra_folder, current_set_images), 'wb') as f:
                pickle.dump(fake_test_list, f, protocol=pickle.HIGHEST_PROTOCOL)
                
            del z_extra
            del fake_test
        G.train()
        print_now('Finished generating extra samples at iteration %d'%((i+1)))

