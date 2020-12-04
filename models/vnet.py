import math
from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .common import conv1_3d, conv1_t_3d,activation,normlayer3d


class VnetBasicBlock(nn.Module):
    def __init__(self, ch, k=5, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        ch1, ch2 = ch

        self.conv1 = conv1_3d((ch1, ch2), k, bias=bias)
        self.norm1 = normlayer3d(ch2,name='batch')
        self.act1 = activation(act)

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'
        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)

    def forward(self, x):
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        return out


class VnetInputTransition(nn.Module):
    def __init__(self,ch,k=5,stride=1,act='prelu',L=1):
        super().__init__()
        self.conv1 = conv1_3d(ch,k,stride)
        self.norm1 = normlayer3d(ch[1],name='batch')
        self.act1 = activation(act)

        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'
        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.norm1(out)
        x16 = th.cat((x, x, x, x, x, x, x, x,
                        x, x, x, x, x, x, x, x), 1)
        out = self.act1(th.add(out,x16))
        return out

class VnetDownwardTransition(nn.Module):
    def __init__(self,chs,nConvs,act='prelu'):
        super().__init__()
        ch1,ch2 = chs
        self.down_conv = conv1_3d((ch1,ch2),k=2,stride=2)
        self.norm1 = normlayer3d(ch2,name='batch')
        self.act1  = activation(act)
        self.ops = _make_nConv((ch2,ch2),nConvs)
        self.act2 = activation(act)

    def forward(self,x):
        down = self.down_conv(x)
        down = self.norm1(down)
        down = self.act1(down)
        out = self.ops(down)
        out = th.add(out,down)
        out = self.act2(out)
        return out

class VnetUpwardTransition(nn.Module):
    def __init__(self,chs,ch_out,nConvs,act='prelu'):
        super().__init__()
        ch1,ch2 = chs
        self.up_conv = conv1_t_3d((ch1,ch2),k=2,stride=2)
        self.norm1 = normlayer3d(ch2,name='batch')
        self.act1  = activation(act)
        self.ops = _make_nConv((ch_out,ch_out),nConvs)
        self.act2 = activation(act)

    def forward(self,x,x_orig):
        up = self.up_conv(x)
        up =  self.norm1(up)
        up = self.act1(up)
        xcat = th.cat((up,x_orig),1)
        out = self.ops(xcat)
        out = th.add(out,xcat)
        out = self.act2(out)
        return out

class VnetOutputtranition(nn.Module):
    def __init__(self,ch,act='sigmoid'):
        super().__init__()
        self.output_block = VnetBasicBlock(ch,act=act)

    def forward(self,x):
        out = self.output_block(x)
        return out

def _make_nConv(ch,depth):
    layers = []
    for _ in range(depth):
        layers.append(VnetBasicBlock(ch,L=depth))
    return nn.Sequential(*layers)

class VNet(nn.Module):

    def __init__(self, layers, k1=7, k=3, cin=1, nk=32, skip=True, act='prelu',
                 down='max3d', up='nearest', G=2, nrm='identity', bias=False,
                 normalize=False, **kwargs):
        super(VNet, self).__init__()

        self.layers = layers
        self.k = k
        self.G = G
        self.ch = (cin, nk)
        self.act = act
        self.skip = skip
        self.down = down
        self.up = up
        self.nrm = nrm
        self.bias = bias
        self.normalize = normalize

        ch1, ch2 = self.ch
        
        self.input_tr = VnetInputTransition((1,16))
        self.down_tr32 = VnetDownwardTransition((16,32),2)
        self.down_tr64 = VnetDownwardTransition((32,64),3)
        self.down_tr128 = VnetDownwardTransition((64,128),3)
        self.down_tr256 = VnetDownwardTransition((128,256),3)
        self.up_tr256 = VnetUpwardTransition((256,128),256,3)
        self.up_tr128 = VnetUpwardTransition((256,64),128,3)
        self.up_tr64 = VnetUpwardTransition((128,32),64,2)
        self.up_tr32 = VnetUpwardTransition((64,16),32,1)
        self.output_tr = VnetOutputtranition((32,1))

        # self.enc = self._make_enc(layers, chs)
        # self.dec = self._make_dec(layers, chs)
        # self.conv2 = conv1_3d((ch2*2, ch1), k=1, bias=bias)

        # # init
        # mo = 'fan_out'
        # nl = 'leaky_relu' if act == 'prelu' else 'relu'

        # nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        # with th.no_grad():
        #     self.conv2.weight /= np.sqrt(np.sum(layers + layers[:-1]))

    # def _make_enc(self, layers, chs):
    #     enc = nn.ModuleList()
    #     for i in range(len(layers)):
    #         layer = []
    #         ch1, ch2 = chs[i]

    #         if i > 0:
    #             d = VnetDownBlock((ch1,ch2))
    #             enc.append(d)
    #             ch1 = ch2

    #         #block = Bottleneck if i > 0 else BasicBlock3D
    #         for j in range(layers[i]):
    #             l = VnetBasicBlock((ch1,ch2),L=layers[i])
    #             layer.append(l)

    #         enc.append(nn.Sequential(*layer))

    #     return enc

    # def _make_dec(self, layers, chs):
    #     dec = nn.ModuleList()

    #     chs = chs[::-1]
    #     layers = layers[-2::-1]
        
    #     ch2 = chs[0][0]
    #     ch1 = chs[0][1]
    #     u = VnetUpBlock((ch1,ch2))
    #     dec.append(u)

    #     for i in range(len(layers)):
    #         layer = []
    #         ch2 = chs[i][1]
    #         ch1 = chs[i][0]

    #         #block = Bottleneck if i > 0 else BasicBlock
    #         for j in range(layers[i]):
    #             ch = (ch2, ch2)
    #             l = VnetBasicBlock(ch,L=layers[i])
    #             layer.append(l)
    #         dec.append(nn.Sequential(*layer))
    #         if i < len(layers)-1:
    #             u = VnetUpBlock((ch2, ch1 // 2))
    #             dec.append(u)

    #     return dec

    def pad(self, x):
        N = len(self.layers)-1
        
        b, c, d, h, w = x.shape

        w1 = ((w - 1) | (2**N - 1)) + 1
        h1 = ((h - 1) | (2**N - 1)) + 1
        d1 = ((d - 1) | (2**N - 1)) + 1

        dw = (w1 - w) / 2
        dh = (h1 - h) / 2
        dd = (d1 - d) / 2
        #print(x.shape)
        if dw == 0 and dh == 0 and dd == 0:
            p = (0, 0, 0, 0, 0, 0)
        else:
            p = math.floor(dw), math.ceil(dw)
            p += math.floor(dh), math.ceil(dh)
            p += math.floor(dd), math.ceil(dd)
            x = F.pad(x, p)
        #print(p)
        return x, p

    def unpad(self, x, p):
        if p != (0, 0, 0, 0 , 0):
            b, c, d, h , w = x.shape
            x = x[..., p[4]:d-p[5], p[2]:h-p[3], p[0]:w-p[1]]
        return x

    def norm(self, x):
        if not self.normalize:
            return x, 0, 1
        else:
            b, c, d, h, w = x.shape
            mu = x.view(b, c, h * w * d).mean(dim=3).view(b, c, 1, 1)
            sigma = x.view(b, c, h * w * d).std(dim=3).view(b, c, 1, 1)
            return (x - mu) / sigma, mu, sigma

    def unnorm(self, x, mu, sigma):
        if not self.normalize:
            return x
        else:
            return x * sigma + mu

    def forward(self, x):
        x, pads = self.pad(x)
        x, mus, sigmas = self.norm(x)

        out16 = self.input_tr(x)
        out_32 = self.down_tr32(out16)
        out_64 = self.down_tr64(out_32)
        out_128 = self.down_tr128(out_64)
        out_256 = self.down_tr256(out_128)
        out = self.up_tr256(out_256,out_128)
        out = self.up_tr128(out,out_64)
        out = self.up_tr64(out,out_32)
        out = self.up_tr32(out,out16)
        out = self.output_tr(out)

        out = self.unpad(out, pads)
        out = self.unnorm(out, mus, sigmas)
        return out

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--layers',
            default = [1, 2, 3, 5, 8],
            nargs = '+',
            type = int,
        )

        parser.add_argument(
            '--cin',
            default = 1,
            type = int
        )

        parser.add_argument(
            '--nk',
            default = 16,
            type = int
        )

        parser.add_argument(
            '--k1',
            default = 7,
            type = int
        )

        parser.add_argument(
            '--k',
            default = 3,
            type = int
        )

        parser.add_argument(
            '--skip',
            default = 1,
            choices = [0, 1],
            type = int
        )

        parser.add_argument(
            '--normalize',
            default = False,
            action = 'store_true'
        )

        parser.add_argument(
            '--act',
            default = 'relu',
            choices = ['relu', 'elu', 'celu', 'selu', 'prelu', 'swish','sigmoid'],
            type = str
        )

        parser.add_argument(
            '--down',
            default = 'max3d',
            choices = ['max', 'max3d', 'mean', 'norm2', 'conv'],
            type = str
        )

        parser.add_argument(
            '--up',
            default = 'nearest',
            choices = ['conv', 'nearest', 'bilinear', 'bicubic'],
            type = str
        )

        parser.add_argument(
            '--nrm',
            default = 'identity',
            choices = ['identity', 'batch', 'instance'],
            type = str
        )

        parser.add_argument(
            '--G',
            default = 2,
            type = int
        )

        parser.add_argument(
            '--bias',
            default = 0,
            choices = [0, 1],
            type = int
        )

        return parser


if __name__ == '__main__':
    th.manual_seed(733)
    model = VNet([1,1,1,2], skip=True)

    th.manual_seed(733)
    x = th.rand(2, 3, 32, 32, 32)

    y = model(x)
