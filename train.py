import os
import torch
import argparse
import numpy as np
from utils.args import *
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from model import BERTLM
from optim_schedule import ScheduledOptim
import scipy.io as sio
import pickle
from data import get_train_loader, get_eval_loader, get_eval_mask_loader
import grad_close
import random
import pandas as pd

import matplotlib.pyplot as plt

from tensorboardX import SummaryWriter

random_seed = 0

torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

if not os.path.exists(file_path):
    os.makedirs(file_path)


best_encoder_optimizer_pth_path = file_path+'/best_encoder_optimizer.pth'
best_decoder_optimizer_pth_path = file_path+'/best_decoder_optimizer.pth'
encoder_optimizer_pth_path = file_path+'/encoder_optimizer.pth'
decoder_optimizer_pth_path = file_path+'/decoder_optimizer.pth'

print('Learning rate: %.4f' % lr)

infos = {}
infos_best = {}
histories = {}
if use_checkpoint is True and os.path.isfile(os.path.join(file_path, 'infos.pkl')):
    with open(os.path.join(file_path, 'infos.pkl')) as f:
        infos = pickle.load(f)

    if os.path.isfile(os.path.join(file_path, 'histories.pkl')):
        with open(os.path.join(file_path, 'histories.pkl')) as f:
            histories = pickle.load(f)

model = BERTLM(feature_size).cuda()
train_loader = get_eval_mask_loader(train_feat_path, batch_size,shuffle = True)

itera = 0
epoch = 0

if use_checkpoint:
    model.load_state_dict(torch.load(file_path + '/9288.pth'))
    itera = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)

encoder_parameters = [{'params': model.bert.parameters()}, {'params': model.binary.parameters()}]
decoder_parameters = [{'params':model.sequence_restore.parameters()}, {'params':model.restore.parameters()}]
optimizer_encoder = Adam(encoder_parameters, lr=lr)
optimizer_decoder = Adam(decoder_parameters, lr=lr)

if os.path.exists(best_encoder_optimizer_pth_path) and use_checkpoint:
    optimizer_encoder.load_state_dict(torch.load(encoder_optimizer_pth_path))
if os.path.exists(best_decoder_optimizer_pth_path) and use_checkpoint:
    optimizer_decoder.load_state_dict(torch.load(decoder_optimizer_pth_path))

def set_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group['lr'] = lr

total_len = len(train_loader)

training_ELBO = []
training_Likelihood = []
training_KL = []

while True:
    decay_factor = 0.9  ** ((epoch)//lr_decay_rate)
    current_lr = max(lr * decay_factor,1e-4)
    set_lr(optimizer_encoder, current_lr)
    set_lr(optimizer_decoder, current_lr)

    ELBO_epoch = 0
    Likelihood_epoch = 0
    KL_epoch = 0

    for i, data in  enumerate(train_loader, start=1):
        optimizer_encoder.zero_grad()
        optimizer_decoder.zero_grad()

        batchsize = data["mask_input"].size(0)
        data = {key: value.cuda() for key, value in data.items()}
        bb1,frame1,hid1,tbar1, prob1 = model.forward(data["mask_input"][:,:max_frames,:])

        #tbar_grad_cal = grad_close.close_form_calculator(model, batch_size = 256, grad_batch_size=1)
        #grad_close_form = tbar_grad_cal.calculate_grad(data["visual_word"][:,:max_frames,:], (tbar1))*torch.sigmoid(tbar1)*torch.sigmoid(-tbar1)

        #tbar_arm_cal = grad_close.arm_calculator(model, sample_num = 10000)
        tbar_u2g_cal = grad_close.u2g_calculator(model, sample_num = 2)

        if epoch > pretrain_epoch - 1:
            #grad_arm = tbar_arm_cal.arm_calculate(data["visual_word"][:,:max_frames,:], tbar1)
            grad_u2g = tbar_u2g_cal.u2g_calculate(data["visual_word"][:,:max_frames,:], tbar1)
            #print('u2g grad:', grad_u2g)


        mask_loss = (torch.sum((frame1-data["visual_word"][:,:max_frames,:])**2))\
                  /(max_frames*feature_size*batchsize)
        KL_unbalanced = torch.sum(prob1*torch.log(torch.div(prob1, prior_prob) + eps) + (1 - prob1)*torch.log(torch.div((1 - prob1), (1 - prior_prob)) + eps))/(nbits*batchsize)
        KL = annealing_rate*KL_unbalanced

        ELBO_epoch = ELBO_epoch + ((mask_loss + KL)*batchsize).detach().cpu().numpy()
        Likelihood_epoch = Likelihood_epoch + ((mask_loss)*batchsize).detach().cpu().numpy()
        KL_epoch = KL_epoch + ((KL_unbalanced)*batchsize).detach().cpu().numpy()

        loss_encoder = mask_loss
        #loss_encoder = 0.92*torch.sum((tbar1*grad_tbar1 + tbar2*grad_tbar2), dim = (0,1)) +0.8*mu_loss
        loss_decoder = mask_loss

        loss_decoder.backward(retain_graph=True)
        optimizer_decoder.step()

        if epoch>pretrain_epoch - 1:
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            tbar1.backward(grad_u2g, retain_graph=True)
        KL.backward(retain_graph=True)
        optimizer_encoder.step()

        itera += 1
        infos['iter'] = itera
        infos['epoch'] = epoch
        if itera%10 == 0 or batchsize<batch_size:  
            print('Epoch:%d Step:[%d/%d] maskloss: %.2f KL: %.2f'  \
            % (epoch, i, total_len,\
               mask_loss.data.cpu().numpy(), \
                    KL.data.cpu().numpy()))

    training_ELBO.append(ELBO_epoch)
    training_Likelihood.append(Likelihood_epoch)
    training_KL.append(KL_epoch)


    torch.save(model.state_dict(), file_path + '/9288.pth')
    torch.save(optimizer_encoder.state_dict(), encoder_optimizer_pth_path)
    torch.save(optimizer_decoder.state_dict(), decoder_optimizer_pth_path)

    with open(os.path.join(file_path, 'infos.pkl'), 'wb') as f:
        pickle.dump(infos, f)
    with open(os.path.join(file_path, 'histories.pkl'), 'wb') as f:
        pickle.dump(histories, f)
    epoch += 1
    if epoch>num_epochs:
        break
    model.train()

#Save training ELBO as json file
pd.DataFrame(training_ELBO).to_csv("trainloss.csv")
pd.DataFrame(training_Likelihood).to_csv("trainlikelihood.csv")
pd.DataFrame(training_KL).to_csv("trainkl.csv")


