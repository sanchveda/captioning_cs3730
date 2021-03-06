

#-----------Pytorch Libraries ---------------#
import numpy as np
from collections import Counter
import h5py
import json
import time
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu

import os 

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

from caption import * 
from models import * 
# computing environment
cuda = torch.cuda.is_available() 
#args.cuda = False

if cuda:
    print('Running on GPU')
    device = torch.device ('cuda:1')
else:
    print('Running on CPU')
    device = torch.device ('cpu')


def test (data_loader, encoder, decoder, criterion):
    """
    Performs an epoch of validation.
    """
    decoder.eval()
    if encoder:
        encoder.eval()
    
  
    ground_truths = []
    ground_truth_unlist = []
    predictions = []
    losses = []
    
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
            
            losses.append (loss)
            #losses.update(loss.item(), sum(decode_lengths))
            #losses.update(loss.item(), sum(decode_lengths))
            #top_accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
          
          
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
                #Put it in list if you have multiple captions ---#
                ground_truths.append([selected_caps])
                ground_truth_unlist.append (selected_caps)
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
            #   convert_to_text (ground_truths[0][0])
            #print (loss)
            
        # use corpus_bleu library functions to calculate bleu score
        bleu_score = corpus_bleu(ground_truths, predictions)
        
        #print(f'\nL {loss.avg:.3f} A {top_5.avg:.3f}, B {bleu_score}\n')
    print (bleu_score)
    
    return bleu_score, predictions, ground_truth_unlist



def evaluate (data_loader, encoder, decoder, criterion):
    """
    Performs an epoch of validation.
    """
    decoder.eval()
    if encoder:
        encoder.eval()
    
  
    ground_truths = []
    ground_truth_unlist = []
    predictions = []

    losses = []
    
    # always disable gradient when evaluating
    with torch.no_grad():
        # all captions also passed in from caption.py
        for i, (imgs, caps, len_caps) in enumerate(data_loader):
            # the uncommented operations are similar to train(), please refer to that
            imgs = imgs.to(device)
            caps = caps.to(device)
            len_caps = len_caps.to(device)
           
            enc_out = encoder(imgs)
            b= enc_out.size (0) #---Batches 
            enc_dim = enc_out.size(-1)
            try:
                enc_out = enc_out.view(b, -1, enc_dim)
                
                num_pix = enc_out.size(1)
            except:
                pdb.set_trace()

            #Now initializing the words 
            prev_words = torch.LongTensor([[word_map['<start>']]] * b).to(device)
            #pdb.set_trace()
            #sorted_caps, decode_lengths, scores, alphas, sorted_idxs = decoder(imgs, caps, len_caps)
            #targets = sorted_caps[:, 1:]
            out_tensor = torch.zeros (b, 500).to(device)
            words = []
            t = 1
            hidden, cell = decoder.init_state(enc_out)
            for i in range(out_tensor.shape[1]) :
                emb = decoder.embedding (prev_words). squeeze(1)
                att_out, _ = decoder.attention (enc_out, emb)

                gate = decoder.sigmoid(decoder.f_beta(hidden))
                att_out = gate * att_out

                hidden, cell = decoder.decode_step(torch.cat([emb, att_out], dim=1), (hidden, cell))
                scores = decoder.fc(hidden)
                scores = F.log_softmax(scores, dim=1)
                _, indices = torch.max(scores, dim=1)

                prev_words = indices

                words.append (indices)
            
            words = torch.stack(words).T
    
            #losses.update(loss.item(), sum(decode_lengths))
            #losses.update(loss.item(), sum(decode_lengths))
            #top_accs.update(accuracy(scores, targets, 5), sum(decode_lengths))
          
          
            # get ground truths (sort captions and get rid of start and end tokens)
            #all_caps = all_caps[sorted_idxs]
        
            prediction = []
            g_truth = []
            for j in range(words.shape[0]):
                img_caps = words[j].tolist()
                g_caps = caps[j].tolist()
                # get rid of <start> and <end> because they increase the bleu score
                selected_caps= []
                for x in img_caps:
                    if x == word_map ['<end>']:
                        break
                    selected_caps.append(x)

                g_caps = [x for x in g_caps if x != word_map['<start>'] and x != word_map['<end>'] and x!=word_map['<pad>']]

                '''
                img_caps = list(map(lambda cap: [w for w in cap if (w != word_map['<start>'] and w != word_map['<pad>'])], 
                                img_caps))
                '''
                prediction.append (selected_caps)
                g_truth.append (g_caps)
                #Put it in list if you have multiple captions ---#
                '''
                ground_truths.append([selected_caps])
                ground_truth_unlist.append (selected_caps)
                '''
            # get predictions
            predictions.extend(prediction)
            ground_truths.extend(g_truth) 

          
            assert len(ground_truths) == len(predictions)
            #   convert_to_text (ground_truths[0][0])
            #print (loss)
            
        # use corpus_bleu library functions to calculate bleu score
        #bleu_score = corpus_bleu(ground_truths, predictions)
        
        #print(f'\nL {loss.avg:.3f} A {top_5.avg:.3f}, B {bleu_score}\n')
    #print (bleu_score)
    
    return predictions, ground_truths

def convert_to_text (token_list):
    new_text = []
    
    for tok in token_list :
        for keys, values in word_map.items():
            
            if values == tok:
                new_text.append (keys)
                
            
 
    new_text = ' '.join(new_text)
  
    return new_text 
def load_checkpoint (path):

    trained_model = torch.load(path)

    epoch = trained_model ['epoch']
    encoder = trained_model ['encoder']
    decoder = trained_model ['decoder']
    encoder_dict = trained_model ['encoder_state_dict']
    decoder_dict = trained_model ['decoder_state_dict']
    encoder_opt = trained_model['encoder_opt']
    decoder_opt = trained_model['decoder_opt']
    bleu = trained_model ['bleu']
    predictions = trained_model ['predictions']
    ground_truth = trained_model ['ground_truth']

    
    return encoder_dict, decoder_dict, encoder_opt, decoder_opt, bleu

input_dir = '/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr_localized_narrative/all_info/' #The information needs to be in a .h5py format
output_dir = "/run/user/1435515045/gvfs/smb-share:server=terracotta.psychology.pitt.edu,share=projects/Sanchayan/flickr_localized_narrative/all_info/model/"

with open(os.path.join(input_dir, "wordmap.json"), 'r') as f:
    word_map = json.load(f)
word_map['<gap>'] = 2055 #This has to be there becauus of a mssing token 

# model
emb_dim = 512       # word embedding dimension
att_dim = 512       # attention dimension/size
dec_dim = 512       # decoder RNN/LSTM dimension
alpha_c = 1.0      
batch_size= 128

model_files = [x for x in os.listdir(output_dir) if x == '12'] [0]  #REPLACE THIS WITH FUULL MODEL PATH

print (model_files) #Seems to be the best score 0.335





encoder_dict, decoder_dict, encoder_opt, decoder_opt, bleu = load_checkpoint (os.path.join (output_dir, model_files))

encoder= Encoder ()
decoder = DecoderWithAttention(len(word_map), emb_dim, dec_dim, att_dim, device=device )
#---Load the weights -----#
encoder.load_state_dict (encoder_dict)
decoder.load_state_dict (decoder_dict)

#decoder_opt = Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()), lr = dec_lr)
decoder = decoder.to(device)
encoder = encoder.to(device)
criterion = nn.CrossEntropyLoss().to(device)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


testset = CapData(input_dir, 'ts', transform=transforms.Compose([normalize]))
 
test_data_loader = data.DataLoader(
    testset,
    batch_size=batch_size, collate_fn= testset.collate_fn, shuffle=False, num_workers=0, pin_memory=False)

test_predictions, test_ground_truth = evaluate(test_data_loader, encoder, decoder, criterion)

best_predictions = test_predictions
best_ground_truth = test_ground_truth
#----------Printing Phase -------------#

for idx, (pred, true) in enumerate(zip(best_predictions, best_ground_truth)):
    
    print (idx, "Pred=", convert_to_text(pred), "Actual=", convert_to_text(true)) 
    
