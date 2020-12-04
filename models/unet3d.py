import math
from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .common import Bottleneck, BasicBlock3DDown,BasicBlock3DUp,BasicBlock3D, DownBlock, UpBlock3D, conv1, conv1_3d


class UNet3D(nn.Module):

    def __init__(self, layers, k1=7, k=3, cin=1, nk=32, skip=True, act='prelu',
                 down='max3d', up='nearest', G=2, nrm='identity', bias=False,
                 normalize=False, **kwargs):
        super(UNet3D, self).__init__()

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
        def fch(i):
            return (round(G**(i)*ch2), round(G**(i+1)*ch2))

        chs = [(ch2, ch2)] + [fch(i) for i in range(len(layers)-1)]
        #chs = [fch(i) for i in range(len(layers))]

        self.conv1 = conv1_3d((ch1, ch2), k=k1, bias=bias)
        self.enc = self._make_enc(layers, chs)
        self.dec = self._make_dec(layers, chs)
        self.conv2 = conv1_3d((ch2, 2), k=1, bias=bias)

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity='conv3d')
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv2.weight /= np.sqrt(np.sum(layers + layers[:-1]))

    def _make_enc(self, layers, chs):
        enc = nn.ModuleList()
        for i in range(len(layers)):
            layer = []
            ch1, ch2 = chs[i]

            if i > 0:
                d = DownBlock(self.down, (ch1, ch1), bias=self.bias)
                layer.append(d)

            #block = Bottleneck if i > 0 else BasicBlock3D
            for j in range(layers[i]):
                ch = (ch1, ch2) if j == 0 else (ch2, ch2)
                l = BasicBlock3DDown(ch, alpha=self.G, k=self.k, act=self.act,
                               nrm=self.nrm, bias=self.bias, skip=self.skip,
                               depth=i, L=layers[i])
                layer.append(l)

            enc.append(nn.Sequential(*layer))

        return enc

    def _make_dec(self, layers, chs):
        dec = nn.ModuleList()

        chs = chs[::-1]
        layers = layers[-2::-1]

        ch2 = chs[0][1]
        u = UpBlock3D(self.up, (ch2, ch2), bias=self.bias)
        dec.append(u)

        for i in range(len(layers)):
            layer = []
            ch2_ = ch2
            ch2 = chs[i+1][1]

            #block = Bottleneck if i > 0 else BasicBlock
            for j in range(layers[i]):
                ch = (ch2_ + ch2, ch2) if j == 0 else (ch2, ch2)
                l = BasicBlock3DUp(ch, k=self.k, alpha=self.G, act=self.act,
                               nrm=self.nrm, bias=self.bias, skip=self.skip,
                               depth=i, L=layers[i])
                layer.append(l)

            if i < len(layers)-1:
                u = UpBlock3D(self.up, (ch2, ch2), bias=self.bias)
                layer.append(u)
            dec.append(nn.Sequential(*layer))

        return dec

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
        identity = x
        x, pads = self.pad(x)
        #print(x.size())
        x, mus, sigmas = self.norm(x)

        u = self.conv1(x)
        u = self.enc[0](u)

        us = []
        for i in range(len(self.enc)-1):
            us.append(u)
            u = self.enc[i+1](u)

        u = self.dec[0](u)
        for i in range(len(self.dec)-1):
            u = th.cat((u, us[-(i+1)]), 1)
            u = self.dec[i+1](u)

        u = self.conv2(u)

        u = self.unpad(u, pads)
        u = self.unnorm(u, mus, sigmas)

        return u + identity

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--layers',
            default = [1, 1, 1, 1, 1],
            nargs = '+',
            type = int,
        )

        parser.add_argument(
            '--cin',
            default = 2,
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
            choices = ['relu', 'elu', 'celu', 'selu', 'prelu', 'swish'],
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
    model = UNet3D([1,1,1,2], skip=True)

    th.manual_seed(733)
    x = th.rand(2, 3, 32, 32, 32)

    y = model(x)
