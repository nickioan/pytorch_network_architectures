from argparse import ArgumentParser

import numpy as np
import torch as th
import torch.nn as nn

from . import add_model_args
from .unet import unet


class VariationalBlock(nn.Module):

    def __init__(self, args, **kwargs):
        super(VariationalBlock, self).__init__()

        self.prox = args.prox
        self.reg = args.reg
        self.tau = th.nn.Parameter(th.ones(1))

        if self.prox:
            self.P = unet()
        else:
            self.P = nn.Identity()

        if self.reg:
            self.R = unet()
        else:
            self.R = None

    def forward(self, x, Y, D):
        P, R, tau = self.P, self.R, self.tau

        X = th.rfft(x, 3, onesided=False, normalized=True)
        DD = D*(D*X - Y)
        DF = th.irfft(DD, 3, onesided=False, normalized=True, signal_sizes=x.size()[-3:])

        if R is None:
            return P(x - tau*DF)
        else:
            return P(x - tau*DF - R(x))


class VariationalNetwork(nn.Module):

    def __init__(self, args, **kwargs):
        super(VariationalNetwork, self).__init__()

        self.niter = args.niter
        self.nunroll = args.nunroll

        if args.initial_guess is None or args.initial_guess == 'zero':
            self.init = vnet()
        else:
            self.init = nn.Identity()

        self.iters = nn.ModuleList([
            VariationalBlock(args) for _ in range(self.nunroll)
        ])

    def forward(self, x, y, D):
        X = th.rfft(x, 3, onesided=False, normalized=True)
        DD = D*X
        DF = th.irfft(DD, 3, onesided=False, normalized=True, signal_sizes=x.size()[-3:])

        u = self.init(DF)
        for _ in range(self.niter):
            for m in self.iters:
                u = m(u, y, D)
        return u

    @staticmethod
    def add_model_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_argument(
            '--varblock',
            default='unet',
            choices = ['unet'],
            type = str,
        )

        parser.add_argument(
            '--nunroll',
            default = 1,
            type = int,
        )

        parser.add_argument(
            '--prox',
            default = False,
            action = 'store_true'
        )

        parser.add_argument(
            '--reg',
            default = False,
            action = 'store_true'
        )

        parser.add_argument(
            '--niter',
            default = 1,
            type = int,
        )

        parser.add_argument(
            '--prox_residual',
            default = 1,
            choices = [0, 1],
            type = int
        )

        tmp_args, _ = parser.parse_known_args()
        return add_model_args(parser, tmp_args.varblock)
