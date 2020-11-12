#mount drive for both input and output
#from google.colab import drive
#drive.mount('/content/drive')

# code
from caption import *
from models	 import *

# miscellaneous
import numpy as np
from collections import Counter
import h5py
import json
import time
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

# PyTorch
import torch
import torch.optim
from torch.optim import Adam
import torch.utils.data as data
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence
import pdb 
from pathlib import Path 
# computing environment
cuda = torch.cuda.is_available() 
#args.cuda = False

if cuda:
    print('Running on GPU')
    device = torch.device ('cuda:1')
else:
    print('Running on CPU')
    device = torch.device ('cpu')
#device= 'cpu'
#cudnn.benchmark = True


# PARAMETERS

# dir
# input_dir = 'drive/My Drive/img_cap/processed_input/' # input directory (processed data)
input_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr_localized_narrative/all_info/'
output_dir = "/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr_localized_narrative/all_info/model/" # save model directory
trained_model = None   # path to pre-trained model, if there is any
Path(output_dir).mkdir(parents=True, exist_ok=True)
# model
emb_dim = 512       # word embedding dimension
att_dim = 512       # attention dimension/size
dec_dim = 512       # decoder RNN/LSTM dimension

# training
enc_lr = 1e-4       # encoder learning rate
dec_lr = 4e-4       # decoder learning rate
batch_size = 16     # batch size, the destroyer of all RAM
grad_threshold = 5. # clip gradient to prevent exploding
alpha_c = 1.        # regularization parameter from the paper
best_bleu = 0.      # highest bleu-4 score
log_freq = 50       # print train/val stats every number of batches
fine_tune = False   # determines whether to fine tune the encoder (resnet 101)

begin_epoch = 0     # beginning epoch (resume training)
num_epoch = 500     # training epochs unless Colab disconnects or early stop
epoch_from_best = 0 # keep track of how many epochs with no improvement

with open(os.path.join(input_dir, "wordmap.json"), 'r') as f:
    word_map = json.load(f)
word_map['<gap>'] = 2055
class StatMeter:
    """
    Keep track of newest val, sum, count, avg.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, repeat=1):
        self.val = val
        self.sum += val * repeat
        self.count += repeat
        self.avg = self.sum / self.count

def lr_decay(opt, factor):
    """
    Perform controlled learning rate decay.
    """
    for pg in opt.param_groups:
        pg['lr'] *= factor

def accuracy(scores, targets, k):
    """
    Compute top-k acc.
    """
    
    batch_size = targets.size(0)
    _, idx = scores.topk(k, dim=1, largest=True, sorted=True)
    
    targets = targets.view(-1, 1).expand_as(idx)
    right = idx.eq(targets)
    
    # sum the right list all up
    right_sum = right.view(-1).float().sum()
    acc = right_sum.item() * (100. / batch_size)
    return acc

def clip_gradient(optimizer, threshold):
	"""
	Clips gradients of every parameter.
	"""

	for pg in optimizer.param_groups:
		for p in pg['params']:
			
			if p.requires_grad:
				p.grad.data.clamp_(-threshold, threshold)

def train(data_loader, encoder, decoder, encoder_opt, decoder_opt, criterion, epoch):
    """
    performs an epoch of training.
    """

    decoder.train()
    encoder.train()
    
    batch_time = StatMeter()   # forward + backward time by batch
    data_time = StatMeter()    # data loading time by batch
    losses = StatMeter()       # loss per word
    top_accs = StatMeter()     # top 5 accuracy
    tick = time.time()         # initialize time
    

    # each batch
    for i, (imgs, caps, len_caps) in enumerate(data_loader):
        data_time.update(time.time() - tick)
       	
        # send them to GPU
        imgs = imgs.to(device)
        caps = caps.to(device)
        len_caps = len_caps.to(device)
        
        # forward: encode images and decode
        imgs = encoder(imgs)
        
        
        sorted_caps, decode_lengths, scores, alphas, sorted_idxs = decoder(imgs, caps, len_caps)
      
     
       	#print (i, sorted_caps.shape)
       
        # get targets (excluding <start>)
        targets = sorted_caps[:, 1:]
       	
        # remove undecoded or simply pads with the PackedSequence object
        scores, _ ,  _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
        targets, _ , _, _= pack_padded_sequence(targets, decode_lengths, batch_first=True)
       
        # compute loss
        loss = criterion(scores, targets)
       	
        # add regularization for soft attention (according to paper)
        loss += alpha_c * ((1. - alphas.sum(dim=1)) ** 2).mean()
        
        # backward (autograd)
        if encoder_opt: # training encoder is very expensive
            encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        loss.backward()
        
        # clip gradients if needed
        if grad_threshold:
            clip_gradient(decoder_opt, grad_threshold)
            if encoder_opt:
                clip_gradient(encoder_opt, grad_threshold)
        
        # update weights with optimizers
        decoder_opt.step()
        if encoder_opt:
            encoder_opt.step()
       
        # update metrics
        losses.update(loss.item(), sum(decode_lengths))
       	
        top_accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
        batch_time.update(time.time() - tick)
        tick = time.time()
        
        '''
        if i % log_freq == 0:
            print(f'E [{epoch}][{i}/{len(data_loader)}]\t'
                  f'D {data_time.val:.3f}->{data_time.avg:.3f}\t'
                  f'B {batch_time.val:.3f}->{batch_time.avg:.3f}\t'
                  f'L {loss.val:.4f}->{loss.avg:.4f}\t'
                  f'A {top_accs.val:.3f}->{top_accs.avg:.3f}')
        '''
        print (loss)
        
   
    
def validate_old(data_loader, encoder, decoder, criterion):
    """
    Performs an epoch of validation.
    """
    decoder.eval()
    if encoder:
        encoder.eval()
    
    # same as train() but without data loading timer
    batch_time = StatMeter()
    losses = StatMeter()
    top_accs = StatMeter()
    tick = time.time()
    
    ground_truths = []
    predictions = []
    
    # always disable gradient when evaluating
    with torch.no_grad():
        # all captions also passed in from caption.py
        for i, (imgs, caps, len_caps, all_caps) in enumerate(data_loader):
            # the uncommented operations are similar to train(), please refer to that
            imgs = imgs.to(device)
            caps = caps.to(device)
            len_caps = len_caps.to(device)
            
            if encoder:
                imgs = encoder(imgs)
            sorted_caps, decode_lengths, scores, alphas, sorted_idxs = decoder(imgs, caps, len_caps)
            targets = sorted_caps[:, 1:]
            
            scores_cp = scores.clone() # save a copy for bleu score
            scores, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. -  alpha.sum(dim=1)) ** 2).mean()
            
            losses.update(loss.item(), sum(decode_lengths))
            losses.update(loss.item(), sum(decode_lengths))
            top_accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
            batch_time.update(time.time() - tick)
            tick = time.time()
            
            if i % log_freq == 0:
                print(f'V [{i}/{len(data_loader)}]\t'
                      f'B {batch_time.val:.3f}->{batch_time.avg:.3f}\t'
                      f'L {loss.val:.4f}->{loss.avg:.4f}\t'
                      f'A {top_accs.val:.3f}->{top_accs.avg:.3f}')

            # get ground truths (sort captions and get rid of start and end tokens)
            all_caps = all_caps[sorted_idxs]
            for j in range(all_caps.shape[0]):
                img_caps = all_caps[j].tolist()
                # get rid of <start> and <end> because they increase the bleu score
                img_caps = list(map(lambda cap: [w for w in cap if (w != word_map['<start>'] and w != word_map['<pad>'])], 
                                img_caps))
                ground_truths.append(img_caps)

            # get predictions
            _, preds = torch.max(scores_cp, dim=2)
            preds = preds.tolist()
            temp = []
            for j, p in enumerate(preds):
                # not including pads
                temp.append(preds[j][:decode_lengths[j]])
            preds = temp
            predictions.extend(preds)
            
            assert len(ground_truths) == len(predictions)
        
        # use corpus_bleu library functions to calculate bleu score
        bleu_score = corpus_bleu(ground_truths, predictions)
        print(f'\nL {loss.avg:.3f} A {top_5.avg:.3f}, B {bleu_score}\n')
        
    return bleu_score

def validate(data_loader, encoder, decoder, criterion):
    """
    Performs an epoch of validation.
    """
    decoder.eval()
    if encoder:
        encoder.eval()
    
    # same as train() but without data loading timer
    batch_time = StatMeter()
    losses = StatMeter()
    top_accs = StatMeter()
    tick = time.time()
    
    ground_truths = []
    predictions = []
    
    # always disable gradient when evaluating
    with torch.no_grad():
        # all captions also passed in from caption.py
        for i, (imgs, caps, len_caps) in enumerate(data_loader):
            # the uncommented operations are similar to train(), please refer to that
            imgs = imgs.to(device)
            caps = caps.to(device)
            len_caps = len_caps.to(device)
            
            if encoder:
                imgs = encoder(imgs)
            sorted_caps, decode_lengths, scores, alphas, sorted_idxs = decoder(imgs, caps, len_caps)
            targets = sorted_caps[:, 1:]
            
            scores_cp = scores.clone() # save a copy for bleu score
            scores, _, _, _ = pack_padded_sequence(scores, decode_lengths, batch_first=True)
            targets, _, _,_ = pack_padded_sequence(targets, decode_lengths, batch_first=True)
            
            loss = criterion(scores, targets)
            loss += alpha_c * ((1. -  alphas.sum(dim=1)) ** 2).mean()
            
            losses.update(loss.item(), sum(decode_lengths))
            losses.update(loss.item(), sum(decode_lengths))
            top_accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
            batch_time.update(time.time() - tick)
            tick = time.time()
            
          
            '''
            if i % log_freq == 0:
                print(f'V [{i}/{len(data_loader)}]\t'
                      f'B {batch_time.val:.3f}->{batch_time.avg:.3f}\t'
                      f'L {loss.val:.4f}->{loss.avg:.4f}\t'
                      f'A {top_accs.val:.3f}->{top_accs.avg:.3f}')
            '''
          
    
            # get ground truths (sort captions and get rid of start and end tokens)
            #all_caps = all_caps[sorted_idxs]
            for j in range(sorted_caps.shape[0]):
                img_caps = sorted_caps[j].tolist()
        
                # get rid of <start> and <end> because they increase the bleu score
                
                selected_caps= [x for x in img_caps if x!=word_map['<start>'] and x!=word_map['<pad>'] and x!=word_map['<end>']]
                '''
                img_caps = list(map(lambda cap: [w for w in cap if (w != word_map['<start>'] and w != word_map['<pad>'])], 
                                img_caps))
                '''
                
                ground_truths.append([selected_caps])

            # get predictions
            _, preds = torch.max(scores_cp, dim=2)
            preds = preds.tolist()
            temp = []
            for j, p in enumerate(preds):
                # not including pads
                temp.append(preds[j][:decode_lengths[j]])
            preds = temp
            predictions.extend(preds)
            
            assert len(ground_truths) == len(predictions)
        
            print (loss)
    
        # use corpus_bleu library functions to calculate bleu score
        bleu_score = corpus_bleu(ground_truths, predictions)
        
        #print(f'\nL {loss.avg:.3f} A {top_5.avg:.3f}, B {bleu_score}\n')
    
    return bleu_score, predictions, ground_truths


def save_model(epoch, encoder, decoder, encoder_opt, decoder_opt, new_bleu, predictions, ground_truth,save_path):

	torch.save ({'epoch' : epoch, 
                'encoder' : encoder, 
                 'decoder' : decoder, 
				'encoder_state_dict': encoder.state_dict(),
                'decoder_state_dict': decoder.state_dict(),
                'encoder_opt': encoder_opt,
                'decoder_opt': decoder_opt, 
                'bleu': new_bleu, 
                'predictions':predictions,
                'ground_truth': ground_truth}, save_path+str(epoch))
        
	return 
def train_and_validate(trained_model=trained_model, best_bleu=best_bleu, begin_epoch=begin_epoch, 
                       epoch_from_best=epoch_from_best, fine_tune=fine_tune, word_map=word_map):
   
    
    decoder = DecoderWithAttention(len(word_map), emb_dim, dec_dim, att_dim, device=device )
    decoder_opt = Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr = dec_lr)
    encoder = Encoder()
    
    encoder.fine_tune(fine_tune)
   	
    if fine_tune:
        encoder_opt = Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr = enc_lr)
    else:
        encoder_opt = None
    
    # move everything to device
    decoder = decoder.to(device)
    encoder = encoder.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    # transform while loading data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
   
   
    trainset = CapData(input_dir, 'tr', transform=transforms.Compose([normalize]))
    validset = CapData(input_dir, 'vl', transform=transforms.Compose([normalize]))
    testset = CapData(input_dir, 'ts', transform=transforms.Compose([normalize]))
    # create data loaders, one worker for moving data, pin_memory for fast data transfer
    train_data_loader = data.DataLoader(
        trainset,
        batch_size=batch_size, collate_fn= trainset.collate_fn,shuffle=True, num_workers=0, pin_memory=False)
    val_data_loader = data.DataLoader(
        validset,
        batch_size=batch_size, collate_fn= validset.collate_fn, shuffle=False, num_workers=0, pin_memory=False)
    test_data_loader = data.DataLoader(
        testset,
        batch_size=batch_size, collate_fn= testset.collate_fn, shuffle=False, num_workers=0, pin_memory=False)
    
    
    for epoch in range(begin_epoch, num_epoch):
        if epoch_from_best >= 25:
            break
        # learning rate decay every 10 epochs with no improvement
        if epoch_from_best % 10 == 0 and epoch_from_best != 0:
            if fine_tune: 
                lr_decay(encoder_opt, 0.8)
            lr_decay(decoder_opt, 0.8)
       	
        # train and validate
        train(train_data_loader, encoder, decoder, encoder_opt, decoder_opt, criterion, epoch)
        val_bleu, val_predictions, val_ground_truth = validate(val_data_loader, encoder, decoder, criterion)
        test_bleu, test_predictions, test_ground_truth = validate(test_data_loader, encoder, decoder, criterion)
        
        update = val_bleu > best_bleu
        
        if update:
            epoch_from_best = 0
            best_bleu = val_bleu
            print("New improvement!")
            save_model(epoch,encoder, decoder, encoder_opt, decoder_opt, best_bleu, test_predictions, test_ground_truth, output_dir)

        else:
            epoch_from_best += 1
            print(f"No improvement for {epoch_from_best} epochs")
        
        
   


train_and_validate()