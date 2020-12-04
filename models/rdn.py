import math
from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .common import DenseConv, DenseBlock, conv1


class RDN(nn.Module):

    def __init__(self, D, C, G, k1=7, k=3, cin=3, nk=64, act='prelu',
                 nrm='identity', bias=False, normalize=False, **kwargs):
        super(RDN, self).__init__()

        self.D = D
        self.C = C
        self.G = G
        self.k = k
        self.ch = (cin, nk)
        self.act = act
        self.nrm = nrm
        self.bias = bias
        self.normalize = normalize

        ch1, ch2 = self.ch

        self.conv1 = conv1((ch1, ch2), k=k1, bias=bias)
        self.conv2 = conv1((ch2, ch2), k=k, bias=bias)

        B = []
        for i in range(self.D):
            B.append(
                DenseBlock((ch2, ch2), G, nconvs=C, k=k, act=act, nrm=nrm, bias=bias, D=D, L=i)
            )
        self.blocks = nn.ModuleList(B)

        self.conv3 = conv1((self.D * ch2, ch2), k=1, bias=bias)
        self.conv4 = conv1((ch2, ch2), k=k, bias=bias)
        self.conv5 = conv1((ch2, ch1), k=1, bias=bias)

    def norm(self, x):
        if not self.normalize:
            return x, 0, 1
        else:
            b, c, h, w = x.shape
            mu = x.view(b, c, h * w).mean(dim=2).view(b, c, 1, 1)
            sigma = x.view(b, c, h * w).std(dim=2).view(b, c, 1, 1)
            return (x - mu) / sigma, mu, sigma

    def unnorm(self, x, mu, sigma):
        if not self.normalize:
            return x
        else:
            return x * sigma + mu

    def forward(self, x):
        identity = x
        x, mus, sigmas = self.norm(x)

        u0 = self.conv1(x)
        u = self.conv2(u0)

        us = []
        for RDB in self.blocks:
            u = RDB(u)
            us.append(u)

        u = self.conv3(th.cat(us, 1))
        u = self.conv4(u)
        u += u0

        u = self.conv5(u)
        u = self.unnorm(u, mus, sigmas)

        return u + identity

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--D',
            default = 16,
            type = int
        )

        parser.add_argument(
            '--C',
            default = 8,
            type = int
        )

        parser.add_argument(
            '--G',
            default = 64,
            type = int
        )

        parser.add_argument(
            '--cin',
            default = 3,
            type = int
        )

        parser.add_argument(
            '--nk',
            default = 64,
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
            '--normalize',
            default = False,
            action = 'store_true'
        )

        parser.add_argument(
            '--act',
            default = 'prelu',
            choices = ['relu', 'elu', 'celu', 'selu', 'prelu', 'swish'],
            type = str
        )

        parser.add_argument(
            '--nrm',
            default = 'identity',
            choices = ['identity', 'batch', 'instance'],
            type = str
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
    model = RDN(2, 2, 8)

    th.manual_seed(733)
    x = th.rand(2, 3, 32, 32)

    y = model(x)
