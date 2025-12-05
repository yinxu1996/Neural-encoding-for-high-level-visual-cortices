import os
import json
import time
import torch as T
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from datasets import *
from utils import *

from torch.utils.model_zoo import load_url as load_state_dict_from_url
from src.torch_training_sequences import *
from src.torch_AlexNet import AlexNet
from src.torch_mpf import Torch_LayerwiseFWRF
from src.torch_fwrf import learn_params_ridge_regression, get_predictions, Torch_fwRF_voxel_block
from src.rf_grid import logspace, model_space_pyramid

import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", type=int, default=1)
#'OFA','FFA','OWFA','VWFA','OPA','EBA','FBA'
parser.add_argument("-brain_region", "--brain_region", type=str, default="OFA")
parser.add_argument("-seed", "--seed", type=int, default=0)
args = parser.parse_args()
sub=args.sub
brain_region = args.brain_region
seed=args.seed

print('\nsub:{} brain_region:{} seed:{}'.format(sub, brain_region, seed))
print("=====================================")

# Data parameters
data_folder = 'dataset/processed_data_article4/subj%02d/'%(sub)
fMRI_info = np.load(data_folder + 'roi_vn.npz')

nv = int(min(list(fMRI_info.values())))

# Model parameters
emb_dim = 512  # dimension of word embeddings
attention_dim = 512  # dimension of attention linear layers
decoder_dim = 512  # dimension of decoder RNN
dropout = 0.5
# device = "cpu"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # sets device for model and PyTorch tensors
cudnn.benchmark = True  # set to true only if inputs to model are fixed size; otherwise lot of computational overhead
torch.manual_seed(seed)
np.random.seed(seed)
fpX = np.float32

# Training parameters
start_epoch = 0
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
holdout_size = 64
workers = 6  # for data-loading; right now, only 1 works with h5py
encoder_lr = 1e-4  # learning rate for encoder if fine-tuning
decoder_lr = 4e-4  # learning rate for decoder
grad_clip = 5.  # clip gradients at an absolute value of
best_val_loss = np.inf
print_freq = 35  # print training/validation stats every __ batches

def main():
    """
    Training.
    """
    global best_val_loss, epochs_since_improvement, start_epoch, brain_region, nv

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    cenercrop = transforms.CenterCrop((227, 227))
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'train', brain_region, transform=transforms.Compose([normalize,cenercrop])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'test', brain_region, transform=transforms.Compose([normalize,cenercrop])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    # 2.encoding + fwRF model
    encoder = AlexNet().to(device)

    # pretrained encoding model
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth', progress=True)
    pre_state_dict = {}
    pre_state_dict['conv1.0.weight'] = state_dict.pop('features.0.weight')
    pre_state_dict['conv1.0.bias'] = state_dict.pop('features.0.bias')
    pre_state_dict['conv2.0.weight'] = state_dict.pop('features.3.weight')
    pre_state_dict['conv2.0.bias'] = state_dict.pop('features.3.bias')
    pre_state_dict['conv3.1.weight'] = state_dict.pop('features.6.weight')
    pre_state_dict['conv3.1.bias'] = state_dict.pop('features.6.bias')
    pre_state_dict['conv4.0.weight'] = state_dict.pop('features.8.weight')
    pre_state_dict['conv4.0.bias'] = state_dict.pop('features.8.bias')
    pre_state_dict['conv5.0.weight'] = state_dict.pop('features.10.weight')
    pre_state_dict['conv5.0.bias'] = state_dict.pop('features.10.bias')
    encoder.load_state_dict(pre_state_dict)

    aperture = fpX(1)
    nx = ny = 11  # x,y
    smin, smax = fpX(0.04), fpX(0.4)  # radius
    ns = 8

    gpf = model_space_pyramid(logspace(ns)(smin, smax), min_spacing=1.4, aperture=1.1*aperture)
    print ('candidate count = ', len(gpf))
    lambdas = np.logspace(3.,7.,9, dtype=fpX)

    for i, (imgs, _, _, _, fMRI) in enumerate(train_loader):
        trn_stim_data = imgs.to(device)
        trn_voxel_data = fMRI.to(device)
        best_losses, best_lambdas, best_params, order = learn_params_ridge_regression(
        trn_stim_data, trn_voxel_data, encoder, gpf, lambdas, \
        aperture=aperture, _nonlinearity=None, zscore=True, sample_batch_size=batch_size, \
        holdout_size=holdout_size, shuffle=True, add_bias=True)

if __name__ == '__main__':
    main()
