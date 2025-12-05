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

# Training parameters
start_epoch = 0
epochs = 100  # number of epochs to train for (if early stopping is not triggered)
epochs_since_improvement = 0  # keeps track of number of epochs since there's been an improvement in validation BLEU
batch_size = 64
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
    for img,_,_,_,_, in test_loader:
        sample_image = img[:20]
        break
    encoder = AlexNet().to(device)
    fmaps = encoder(sample_image[:20].to(device))
    fwrf = Torch_LayerwiseFWRF(fmaps, nv=nv, pre_nl=None, post_nl=None, dtype=np.float32).to(device)
    
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
    
    opts = torch.optim.Adam([
        {'params': fwrf.parameters()},
    ], lr=1e-3, betas=(0.9, 0.999), eps=1e-08)
  
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epoch == 11:
            break
        if epochs_since_improvement > 0:
            adjust_learning_rate(opts, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              fwrf = fwrf,
              optimizer=opts,
              epoch=epoch)
        
        # One epoch's test
        val_loss = validate(val_loader=test_loader,
                            encoder=encoder,
                            fwrf=fwrf)

        # Check if there was an improvement
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_AlexNet_RF_checkpoint(epoch, epochs_since_improvement, encoder, fwrf, opts,val_loss, is_best, brain_region,
                                seed, data_folder)

def train(train_loader,encoder,fwrf,optimizer,epoch):
    
    encoder.train()
    fwrf.train()  # train mode (dropout and batchnorm is used)

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    msees = AverageMeter()
    cces = AverageMeter()
    start = time.time()

    # Batches
    for i, (imgs, _, _, _, fMRI) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        fMRI = fMRI.to(device)

        # Forward prop.
        imgs = encoder(imgs)
        vr = fwrf(imgs)[0]

        # Calculate loss
        loss = vox_loss_fn(vr, fMRI)
        loss += 0.1 * T.sum(T.abs(fwrf.w))

        # Back prop.
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)
        # Update weights
        optimizer.step()

        # Keep track of metrics
        mse, cc = predfMRI_metrics(vr, fMRI)
        losses.update(loss.item())
        msees.update(mse.mean())
        cces.update(cc.mean())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data Load Time {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MSE {mse.val:.3f} ({mse.avg:.3f})\t'
                  'cc {cc.val:.3f} ({cc.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses,
                                                            mse=msees, cc=cces))


def validate(val_loader, encoder, fwrf):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    encoder.eval()  # eval mode (no dropout or batchnorm)
    fwrf.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    msees = AverageMeter()
    cces = AverageMeter()
    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, _, _, _, fMRI) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            fMRI = fMRI.to(device)

            # Forward prop.
            imgs = encoder(imgs)
            vr = fwrf(imgs)[0]
            
            # Calculate loss
            loss = vox_loss_fn(vr, fMRI)
            loss += 0.1 * T.sum(T.abs(fwrf.w))

            # Keep track of metrics
            mse, cc = predfMRI_metrics(vr, fMRI)
            losses.update(loss.item())
            msees.update(mse.mean())
            cces.update(cc.mean())
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'MSE {mse.val:.3f} ({mse.avg:.3f})\t'
                      'cc {cc.val:.3f} ({cc.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                loss=losses, mse=msees, cc=cces))
        print('\n * MSE - {mse.avg:.3f}, cc - {cc.avg:.3f}, \n'.format(
                mse=msees, cc=cces))

    return losses.avg

if __name__ == '__main__':
    main()
