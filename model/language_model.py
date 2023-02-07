import torch
import torch.nn as nn
from torch.autograd import Variable, Function
from .bert import BERT
from .args import nbits,hidden_size
#from .globalvar import *

class Round3(Function):
    @staticmethod
    def forward(ctx, input, training=False, inplace=False):
        output = torch.round(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        mask = ~(ctx.input==0)
        mask = Variable(mask).cuda().float()
        grad_output = grad_output*mask
        return grad_output, None, None

class Bernoulli_sample(Function):
    @staticmethod
    def forward(ctx, input):
        output = torch.bernoulli(input)
        ctx.input = input
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None

class BERTLM(nn.Module):
    """
    BERT Language Model
    Pooling and restore: Adding a parameter.
    Method Sampling:
    """

    def __init__(self, frame_size):

        super(BERTLM,self).__init__()
        self.bert = BERT(frame_size, hidden=hidden_size, n_layers=1, attn_heads=1, dropout=0.1)
        self.restore = nn.Linear(nbits, frame_size)
        self.sequence_restore = nn.Linear(1, 25, bias=False)
        self.binary = nn.Linear(hidden_size, nbits)
        self.activation = self.binary_tanh_unit

        self.sigmoid = nn.Sigmoid()

    def hard_sigmoid(self,x):
        y = (x+1.)/2.
        y[y>1] = 1
        y[y<0] = 0
        return y

    def binary_tanh_unit(self,x):
        #This is similar to hyperbolic tangent function but not exactly the same.
        #It looks like here they are still using y = x(-1<x<1)to approximate sgn function.
        #In forward pass, out = sgn(x). In backward pass, dout/dx = 1(-1<x<1)
        #Also, they calculate backward in-place. Why?
        #Why the mask is need here? Why they do mask operation here to "input==0" but not to "input == 1"? Need to check carefully.
        y = self.hard_sigmoid(x)
        out = 2.*Round3.apply(y)-1.
        return out

    def bernoullisampling(self, phi):
        p = self.sigmoid(phi)
        #bb = torch.bernoulli(p)
        bb = 2.*Bernoulli_sample.apply(p)-1
        return bb, p

    def forward(self, x):
        hid = self.bert(x)  #Hid is the latent representation
        z = self.binary(hid)
        tbar = torch.mean(z, 1)
        bb, prob = self.bernoullisampling(tbar)
        sequence_bb = self.sequence_restore(bb.unsqueeze(2)).transpose(1, 2)
        frame = self.restore(sequence_bb)
        return bb, frame, hid, tbar, prob

    def forward_determinstic(self, x):
        hid = self.bert(x)  #Hid is the latent representation
        z = self.binary(hid)
        tbar = torch.mean(z, 1)
        bb, prob = self.bernoullisampling(tbar)
        sequence_bb = self.sequence_restore(bb.unsqueeze(2)).transpose(1, 2)
        return prob

    def encoder_forward(self, x):
        hid = self.bert(x)  #Hid is the latent representation
        z = self.binary(hid)
        tbar = torch.mean(z, 1)
        return tbar

    def decoder_forward(self, bb):
        sequence_bb = self.sequence_restore(bb.unsqueeze(2)).transpose(1, 2)
        frame = self.restore(sequence_bb)
        return frame
