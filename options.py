import argparse

def strToBool(str):
    return str.lower() in ('true', 'yes', 'on', 't', '1')

def get_args():
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', strToBool)
    
    # base options
    parser.add_argument('--CIFAR10', type='bool', default=False, help='If True, use CIFAR-10 instead of your own dataset.')
    parser.add_argument('--input_folder', default='path/to/dataset', help='Input folder.')
    parser.add_argument('--output_folder', default='path/to/output', help='Path to save model and training snapshots.')
    parser.add_argument('--extra_folder', default='path/to/extra_images', help='Path to store extra images.')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--effective_batch_size', type=int, default=32, help='Actual batch size when backpropogating.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size when loading images.') 
    parser.add_argument('--image_size', type=int, default=256)
    parser.add_argument('--n_channels', type=int, default=3, help='Number of color channels.')
    
    parser.add_argument('--z_size', type=int, default=128, help='Dimension of latent input.')
    parser.add_argument('--G_h_size', type=int, default=128, help='Number of hidden nodes in G.')
    parser.add_argument('--D_h_size', type=int, default=128, help='Number of hidden nodes in D.') 
    parser.add_argument('--lr_G', type=float, default=.0001, help='Generator learning rate.')
    parser.add_argument('--lr_D', type=float, default=.0001, help='Discriminator learning rate.')
    parser.add_argument('--total_iters', type=int, default=100000, help='Number of iteration cycles.')
    parser.add_argument('--D_updates', type=int, default=1, help='Number of D updating per iteration cycle.')
    parser.add_argument('--G_updates', type=int, default=1, help='Number of G updating per iteration cycle.')
    
    parser.add_argument('--adam_eps', type=float, default=1e-08, help='Adam eps.')
    parser.add_argument('--beta1', type=float, default=0.5, help='Adam betas[0].')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam betas[1].')
    parser.add_argument('--decay', type=float, default=0, help='Decay to apply to lr each cycle. decay^n_iter gives the final lr.')
    parser.add_argument('--weight_decay', type=float, default=0, help='L2 regularization weight. Helps convergence but leads to artifacts in images, not recommended.')

    parser.add_argument('--cuda', type='bool', default=True, help='Enable cuda.')
    parser.add_argument('--n_gpu', type=int, default=1, help='Number of GPUs to use.')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers in DataLoader.')

    parser.add_argument('--gen_extra_images', type=int, default=50000, help='Generate additional images for evaluation (FID/SWD). Must be a multiple of 100.')
    parser.add_argument('--gen_every', type=int, default=100000, help='Generate additional images every X iterations.')
    parser.add_argument('--print_every', type=int, default=1000, help='Generate a mini-batch of images every X iterations (to see how the training progress, you can do it often).')

    parser.add_argument('--load_ckpt', default=None, help='Path to load checkpoint.')

    # RealnessGAN 
    parser.add_argument('--positive_skew', type=float, default=1.0, help='Skewness of anchor1 when computing loss.')
    parser.add_argument('--negative_skew', type=float, default=-1.0, help='Skewness of anchor0 when computing loss.')
    parser.add_argument('--num_outcomes', type=int, default=20, help='Number of outcomes of D.')
    parser.add_argument('--relativisticG', type='bool', default=True, help='Whether to use relativistic trick when training G.')
    parser.add_argument('--use_adaptive_reparam', type='bool', default=True, help='Whether to use re-parameterization trick in training.')

    param = parser.parse_args()

    return param







    
    




