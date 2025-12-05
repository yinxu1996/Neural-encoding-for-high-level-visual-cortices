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
from models import Encoder, DecoderWithAttention, PredFMRIDecoder
from datasets import *
from utils import *
from nltk.translate.bleu_score import corpus_bleu
import warnings
warnings.filterwarnings("ignore")

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", type=int, default=1)
#'OFA','FFA','OWFA','VWFA','OPA','EBA','FBA'
parser.add_argument("-brain_region", "--brain_region", type=str, default="OFA")
parser.add_argument("-seed", "--seed", type=int, default=0)
parser.add_argument("-fine_tune_encoder", "--fine_tune_encoder", type=bool, default=True)
args = parser.parse_args()
sub=args.sub
brain_region = args.brain_region
seed=args.seed
fine_tune_encoder = args.fine_tune_encoder


print('\nsub:{} brain_region:{} seed:{} fine_tune_encoder:{}'.format(sub, brain_region, seed, fine_tune_encoder))
print("===============================================")

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
batch_size = 128
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
    global best_val_loss, epochs_since_improvement, start_epoch, fine_tune_encoder, \
        word_map, brain_region, nv

    # Read word map
    word_map_file = os.path.join(data_folder, 'WORDMAP.json')
    with open(word_map_file, 'r') as j:
        word_map = json.load(j)

    predfMRI_decoder = PredFMRIDecoder(nv=nv, channel=2048, size=14, init_value=0, pre_nl=True, post_nl=True)
    predfMRI_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, predfMRI_decoder.parameters()),
                                                  lr=decoder_lr)
    
    ICM_decoder = DecoderWithAttention(attention_dim=attention_dim,
                                        embed_dim=emb_dim,
                                        decoder_dim=decoder_dim,
                                        vocab_size=len(word_map),
                                        dropout=dropout)
    ICM_decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, ICM_decoder.parameters()),
                                                lr=decoder_lr)
    encoder = Encoder()
    encoder.fine_tune(fine_tune_encoder)
    encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()),
                                            lr=encoder_lr) if fine_tune_encoder else None


    # Move to GPU, if available
    predfMRI_decoder = predfMRI_decoder.to(device)
    ICM_decoder = ICM_decoder.to(device)
    encoder = encoder.to(device)

    # Loss function
    ICM_criterion = nn.CrossEntropyLoss().to(device)
    predfMRI_criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    train_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'train', brain_region, transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        CaptionDataset(data_folder, 'test', brain_region, transform=transforms.Compose([normalize])),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    
    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 3:
            break
        if epochs_since_improvement > 0:
            adjust_learning_rate(ICM_decoder_optimizer, 0.8)
            if fine_tune_encoder:
                adjust_learning_rate(encoder_optimizer, 0.8)

        # One epoch's training
        train(train_loader=train_loader,
              encoder=encoder,
              ICM_decoder=ICM_decoder,
              predfMRI_decoder = predfMRI_decoder,
              ICM_criterion=ICM_criterion,
              predfMRI_criterion = predfMRI_criterion,
              encoder_optimizer=encoder_optimizer,
              ICM_decoder_optimizer=ICM_decoder_optimizer,
              predfMRI_decoder_optimizer=predfMRI_decoder_optimizer,
              epoch=epoch)
        
        # One epoch's test
        val_loss = validate(val_loader=test_loader,
                            encoder=encoder,
                            ICM_decoder=ICM_decoder,
                            predfMRI_decoder = predfMRI_decoder,
                            ICM_criterion=ICM_criterion,
                            predfMRI_criterion = predfMRI_criterion)

        # Check if there was an improvement
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, epochs_since_improvement, encoder, ICM_decoder, predfMRI_decoder, encoder_optimizer,
                        ICM_decoder_optimizer, predfMRI_decoder_optimizer, val_loss, is_best, brain_region,
                        seed, data_folder, fine_tune_encoder)


def train(train_loader, encoder, ICM_decoder, predfMRI_decoder, ICM_criterion, predfMRI_criterion, 
          encoder_optimizer, ICM_decoder_optimizer, predfMRI_decoder_optimizer, epoch):
    
    predfMRI_decoder.train()
    ICM_decoder.train()  # train mode (dropout and batchnorm is used)
    encoder.train()

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss (per word decoded)
    top5accs = AverageMeter()  # top5 accuracy
    msees = AverageMeter()
    cces = AverageMeter()
    start = time.time()

    # Batches
    for i, (imgs, caps, caplens, _, fMRI) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to GPU, if available
        imgs = imgs.to(device)
        caps = caps.to(device)
        caplens = caplens.to(device)
        fMRI = fMRI.to(device)

        # Forward prop.
        imgs = encoder(imgs)  # batchsize,14,14,2048
        scores, caps_sorted, decode_lengths, alphas, sort_ind = ICM_decoder(imgs, caps, caplens)
        rf_weights = T.mean(alphas, dim=(0,1))
        vr, Phi = predfMRI_decoder(imgs, rf_weights)  # batchsize,nv  nv,batchsize,feature 

        # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
        targets = caps_sorted[:, 1:]
        # Remove timesteps that we didn't decode at, or are pads
        # pack_padded_sequence is an easy trick to do this
        scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]

        # Calculate loss
        ICM_loss = ICM_criterion(scores, targets) + ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        rf = predfMRI_decoder.rf.unsqueeze(1)
        laplacian_rf = F.conv2d(rf, laplacian, padding=1)
        predfMRI_loss = predfMRI_criterion(vr, fMRI) + T.abs(rf).mean() + (laplacian_rf**2).mean()
        
        # Add doubly stochastic attention regularization
        loss = ICM_loss + predfMRI_loss

        # Back prop.
        ICM_decoder_optimizer.zero_grad()
        predfMRI_decoder_optimizer.zero_grad()
        if encoder_optimizer is not None:
            encoder_optimizer.zero_grad()
        loss.backward()

        # Clip gradients
        if grad_clip is not None:
            clip_gradient(ICM_decoder_optimizer, grad_clip)
            clip_gradient(predfMRI_decoder_optimizer, grad_clip)
            if encoder_optimizer is not None:
                clip_gradient(encoder_optimizer, grad_clip)

        # Update weights
        ICM_decoder_optimizer.step()
        predfMRI_decoder_optimizer.step()
        if encoder_optimizer is not None:
            encoder_optimizer.step()

        # Keep track of metrics
        top5 = ICM_accuracy(scores, targets, 5)
        mse, cc = predfMRI_metrics(vr, fMRI)
        losses.update(loss.item(), sum(decode_lengths))
        top5accs.update(top5, sum(decode_lengths))
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
                  'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                  'MSE {mse.val:.3f} ({mse.avg:.3f})\t'
                  'cc {cc.val:.3f} ({cc.avg:.3f})\t'.format(epoch, i, len(train_loader),
                                                            batch_time=batch_time,
                                                            data_time=data_time, loss=losses,
                                                            top5=top5accs, mse=msees,
                                                            cc=cces))


def validate(val_loader, encoder, ICM_decoder, predfMRI_decoder, ICM_criterion, predfMRI_criterion):
    """
    Performs one epoch's validation.

    :param val_loader: DataLoader for validation data.
    :param encoder: encoder model
    :param decoder: decoder model
    :param criterion: loss layer
    :return: BLEU-4 score
    """
    ICM_decoder.eval()  # eval mode (no dropout or batchnorm)
    predfMRI_decoder.eval()
    if encoder is not None:
        encoder.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top5accs = AverageMeter()
    msees = AverageMeter()
    cces = AverageMeter()
    start = time.time()

    # explicitly disable gradient calculation to avoid CUDA memory error
    # solves the issue #57
    with torch.no_grad():
        # Batches
        for i, (imgs, caps, caplens, _, fMRI) in enumerate(val_loader):

            # Move to device, if available
            imgs = imgs.to(device)
            caps = caps.to(device)
            caplens = caplens.to(device)
            fMRI = fMRI.to(device)

            # Forward prop.
            if encoder is not None:
                imgs = encoder(imgs)
            scores, caps_sorted, decode_lengths, alphas, sort_ind = ICM_decoder(imgs, caps, caplens)
            rf_weights = T.mean(alphas, dim=(0,1))
            vr, Phi = predfMRI_decoder(imgs, rf_weights)  # batchsize,nv  nv,batchsize,feature

            # Since we decoded starting with <start>, the targets are all words after <start>, up to <end>
            targets = caps_sorted[:, 1:]

            # Remove timesteps that we didn't decode at, or are pads
            # pack_padded_sequence is an easy trick to do this
            scores_copy = scores.clone()
            scores = pack_padded_sequence(scores, decode_lengths, batch_first=True)[0]
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True)[0]
            
            # Calculate loss
            ICM_loss = ICM_criterion(scores, targets) + ((1. - alphas.sum(dim=1)) ** 2).mean()
            
            laplacian = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            rf = predfMRI_decoder.rf.unsqueeze(1)
            laplacian_rf = F.conv2d(rf, laplacian, padding=1)
            predfMRI_loss = predfMRI_criterion(vr, fMRI) + T.abs(rf).mean() + (laplacian_rf**2).mean()

            # Add doubly stochastic attention regularization
            loss = ICM_loss + predfMRI_loss

            # Keep track of metrics
            top5 = ICM_accuracy(scores, targets, 5)
            mse, cc = predfMRI_metrics(vr, fMRI)
            losses.update(loss.item(), sum(decode_lengths))
            top5accs.update(top5, sum(decode_lengths))
            msees.update(mse.mean())
            cces.update(cc.mean())
            batch_time.update(time.time() - start)

            start = time.time()

            if i % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top-5 Accuracy {top5.val:.3f} ({top5.avg:.3f})\t'
                      'MSE {mse.val:.3f} ({mse.avg:.3f})\t'
                      'cc {cc.val:.3f} ({cc.avg:.3f})\t'.format(i, len(val_loader), batch_time=batch_time,
                                                                loss=losses, top5=top5accs, mse=msees,
                                                                cc=cces))
        print('\n * MSE - {mse.avg:.3f}, cc - {cc.avg:.3f}, TOP-5 ACCURACY - {top5.avg:.3f}, \n'.format(
                mse=msees, cc=cces, top5=top5accs))

    return losses.avg


if __name__ == '__main__':
    main()
