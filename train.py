import os
import numpy as np
import random
import math
import gc
import pickle
import time
import sys
import datetime
from scipy.stats import skewnorm

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
title = 'SEED-%d' % param.seed
base_dir = f"{param.output_folder}/{title}"
if param.load_ckpt is None and os.path.exists(base_dir):
    print_now('Base dir exists! Please check {}.'.format(base_dir))
    raise ValueError

if param.load_ckpt is None:
    os.system('mkdir -p %s' % base_dir)
    os.mkdir(f"{base_dir}/images")
    if param.gen_extra_images > 0 and not os.path.exists(f"{param.extra_folder}"):
        os.mkdir(f"{param.extra_folder}")
    print_now(param)

if param.cuda:
    import torch.backends.cudnn as cudnn
    cudnn.deterministic = True
    device = 'cuda'

to_img = transf.ToPILImage()
torch.utils.backcompat.broadcast_warning.enabled=True

random.seed(param.seed)
numpy.random.seed(param.seed)
torch.manual_seed(param.seed)
if param.cuda:
    torch.cuda.manual_seed_all(param.seed)

trans = transf.Compose([
    transf.Resize((param.image_size, param.image_size)),
    transf.ToTensor(),
    transf.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])
])

if param.CIFAR10:
    data = dset.CIFAR10(root=param.input_folder, train=True, download=False, transform=trans)
else:
    data = dset.ImageFolder(root=param.input_folder, transform=trans)

class DataProvider:
    def __init__(self, data, batch_size):
        self.data_loader = None
        self.iter = None
        self.batch_size = batch_size
        self.data = data
        self.data_loader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=param.num_workers)
        self.build()
    def build(self):
        self.iter = iter(self.data_loader) 
    def __next__(self):
        try:
            return self.iter.next()
        except StopIteration:  # reload when an epoch finishes
            self.build()
            return self.iter.next()

random_sample = DataProvider(data, param.batch_size)

from model.DCGAN_model import DCGAN_G, DCGAN_D
G = DCGAN_G(param)
D = DCGAN_D(param)
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

print_now('Using image size {} with output channel = {}'.format(param.image_size, param.n_channels))
print_now('Using {} GPUs, batch_size = {}, effective_batch_size = {}'.format(param.n_gpu, param.batch_size, param.effective_batch_size))

# used for inspecting training process
with open('./test_z_vec/z_test_%d_%d.pickle'%(param.batch_size, param.z_size), 'rb') as f:
    z_test = pickle.load(f)

# to cuda
G = G.to(device)
D = D.to(device)
Triplet_Loss.to(device)
z_test = z_test.to(device)
x = torch.FloatTensor(param.batch_size, param.n_channels, param.image_size, param.image_size).to(device)
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
    z_test.copy_(checkpoint['z_test'])
    del checkpoint
    print_now(f'Resumed from iteration {current_set_images * param.gen_every}.')
else:
    current_set_images = 0
    iter_offset = 0

print_now(G)
print_now(D)

# define anchors (you can experiment different shapes)

# e.g. skewed normals
# skew = skewnorm.rvs(-5, size=1000)
# count, bins = np.histogram(skew, param.num_outcomes)
# anchor0 = count / sum(count)

# skew = skewnorm.rvs(5, size=1000)
# count, bins = np.histogram(skew, param.num_outcomes)
# anchor1 = count / sum(count)

# e.g. normal and uniform
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

    # save training progress
    if (i+1) % param.print_every == 0:
        G.eval()
        with torch.no_grad():
            fake_test = G(z_test)
            vutils.save_image(fake_test.data, '%s/images/fake_samples_iter%05d.png' % (base_dir, i+1), normalize=True)
        G.train()

    lossD = learnD_Realness(param, D, G, optimizerD, random_sample, Triplet_Loss, x, anchor1, anchor0)
    lossG = learnG_Realness(param, D, G, optimizerG, random_sample, Triplet_Loss, x, anchor1)

    decayD.step()
    decayG.step()

    # print progress
    if i < 1000 or (i+1) % 100 == 0:
        end = time.time()
        fmt = '[%d / %d] SD: %d Diff: %.4f loss_D: %.4f loss_G: %.4f time:%.2f'
        s = fmt % (i+1, param.total_iters, param.seed,
                    -lossD.data.item() + lossG.data.item() if (lossD is not None) and (lossG is not None) else -1.0,
                    lossD.data.item()                      if lossD is not None else -1.0,
                    lossG.data.item()                      if lossG is not None else -1.0,
                    end - start)
        print_now(s)

    # generate extra images
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
            'z_test': z_test,
        }, '%s/models/state_%02d.pth' % (param.extra_folder, current_set_images))
        print_now('Model saved.')

        if os.path.exists('%s/%01d/' % (param.extra_folder, current_set_images)):
            for root, dirs, files in os.walk('%s/%01d/' % (param.extra_folder, current_set_images)):
                for f in files:
                    os.unlink(os.path.join(root, f))
        else:
            os.mkdir('%s/%01d/' % (param.extra_folder, current_set_images))

        G.eval()
        extra_batch = 100 if param.image_size <= 256 else param.batch_size
        with torch.no_grad():
            ext_curr = 0
            z_extra = torch.FloatTensor(extra_batch, param.z_size, 1, 1)
            z_extra = z_extra.to(device)
            for ext in range(int(param.gen_extra_images/extra_batch)):
                fake_test = G(z_extra.normal_(0, 1)) 
                    
                for ext_i in range(fake_test.size(0)):
                    vutils.save_image((fake_test[ext_i]*.50)+.50, '%s/%01d/fake_samples_%05d.png' % (param.extra_folder, current_set_images, ext_curr),
                        normalize=False, padding=0)
                    ext_curr += 1
            del z_extra
            del fake_test
        G.train()
        print_now('Finished generating extra samples at iteration %d'%((i+1)))
        







    
    

    






