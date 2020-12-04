import numpy as np
import torch as th
import torch.nn as nn

def conv1_3d(ch, k, stride=1, bias=False, padding='zeros'):
    return nn.Conv3d(ch[0], ch[1], k, stride=stride, padding=(k-1)//2,
                     bias=bias, padding_mode=padding)

def conv1_t_3d(ch, k, stride=1, bias=False):
    return nn.ConvTranspose3d(ch[0], ch[1], k, stride=stride, padding=(k-1)//2,
                              bias=bias)

def conv1(ch, k, stride=1, bias=False, padding='zeros'):
    return nn.Conv2d(ch[0], ch[1], k, stride=stride, padding=(k-1)//2,
                     bias=bias, padding_mode=padding)


def conv1_t(ch, k, stride=1, bias=False):
    return nn.ConvTranspose2d(ch[0], ch[1], k, stride=stride, padding=(k-1)//2,
                              bias=bias)


class DownBlock(nn.Module):

    def __init__(self, mode='max3d', ch=None, bias=False):
        super().__init__()

        if ch is not None and ch[0] != ch[1]:
            raise ValueError('input, output channels must be equal')

        if mode == 'max':
            self.down = nn.MaxPool2d(2, stride=2)
        elif mode == 'max3d':
            self.down = nn.MaxPool3d(2,stride = 2)
        elif mode == 'mean':
            self.down = nn.AvgPool2d(2, stride=2)
        elif mode == 'norm2':
            self.down = nn.LPPool2d(2, 2, stride=2)
        elif mode == 'conv':
            self.down = conv1(ch, k=2, stride=2, bias=bias)
        else:
            raise ValueError("mode must be one of maxpool, norm2, conv")

    def forward(self, x):
        #print(x.shape)
        return self.down(x)


class UpBlock(nn.Module):

    def __init__(self, mode='nearest', ch=None, bias=False):
        super().__init__()

        if ch is not None and ch[0] != ch[1]:
            raise ValueError('input, output channels must be equal')

        if mode == 'conv':
            self.up = conv1_t(ch, k=2, stride=2, bias=bias)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)

    def forward(self, x):
        return self.up(x)

class UpBlock3D(nn.Module):

    def __init__(self, mode='nearest', ch=None, bias=False):
        super().__init__()

        if ch is not None and ch[0] != ch[1]:
            raise ValueError('input, output channels must be equal')

        if mode == 'conv':
            self.up = conv1_t_3d(ch, k=2, stride=2, bias=bias)
        elif mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode=mode)
        else:
            self.up = nn.Upsample(scale_factor=2, mode=mode, align_corners=True)

    def forward(self, x):
        #print(x.shape)
        return self.up(x)


class BasicBlock(nn.Module):

    def __init__(self, ch, k=3, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        if nrm is None:
            nrm = 'identity'

        ch1, ch2 = ch

        if ch1 == ch2:
            self.bottle = nn.Identity()
        else:
            self.bottle = conv1((ch1, ch2), k=1, bias=bias)

        self.conv1 = conv1((ch2, ch2), k, bias=bias)
        self.norm1 = normlayer(ch2, nrm)
        self.act1 = activation(act)

        self.conv2 = conv1((ch2, ch2), k, bias=bias)
        self.norm2 = normlayer(ch2, nrm)
        self.act2 = activation(act)

        #self.se = SEBlock(ch2)

        self.skip = skip

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        if ch1 != ch2:
            _nl = 'conv2d' if ch2 > ch1 else nl
            nn.init.kaiming_uniform_(self.bottle.weight, mode=mo, nonlinearity=_nl)

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)
            self.conv2.weight /= np.sqrt(L)
            if isinstance(self.bottle, nn.Conv2d):
                self.bottle.weight /= np.sqrt(depth+1)

    def forward(self, x):
        out = self.bottle(x)
        identity = out

        #out = self.se(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        if self.skip:
            out += identity

        return out

class BasicBlock3D(nn.Module):

    def __init__(self, ch, k=3, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        if nrm is None:
            nrm = 'identity'

        ch1, ch2 = ch

        if ch1 == ch2:
            self.bottle = nn.Identity()
        else:
            self.bottle = conv1_3d((ch1, ch2), k=1, bias=bias)

        self.conv1 = conv1_3d((ch2, ch2), k, bias=bias)
        self.norm1 = normlayer3d(ch2, nrm)
        self.act1 = activation(act)

        self.conv2 = conv1_3d((ch2, ch2), k, bias=bias)
        self.norm2 = normlayer3d(ch2, nrm)
        self.act2 = activation(act)

        #self.se = SEBlock(ch2)

        self.skip = skip

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        if ch1 != ch2:
            _nl = 'conv3d' if ch2 > ch1 else nl
            nn.init.kaiming_uniform_(self.bottle.weight, mode=mo, nonlinearity=_nl)

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)
            self.conv2.weight /= np.sqrt(L)
            if isinstance(self.bottle, nn.Conv3d):
                self.bottle.weight /= np.sqrt(depth+1)

    def forward(self, x):
        #print(x.shape)
        out = self.bottle(x)
        #print(out.shape)
        identity = out

        #out = self.se(out)

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        if self.skip:
            out += identity

        return out


class BasicBlock3DDown(nn.Module):

    def __init__(self, ch, k=3, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        if nrm is None:
            nrm = 'identity'

        ch1, ch2 = ch

        self.conv1 = conv1_3d((ch1, ch1), k, bias=bias)
        self.norm1 = normlayer3d(ch1, nrm)
        self.act1 = activation(act)

        self.conv2 = conv1_3d((ch1, ch2), k, bias=bias)
        self.norm2 = normlayer3d(ch2, nrm)
        self.act2 = activation(act)

        #self.se = SEBlock(ch2)

        self.skip = skip

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)
            self.conv2.weight /= np.sqrt(L)

    def forward(self, x):
        #out = self.se(out)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        return out

class BasicBlock3DUp(nn.Module):

    def __init__(self, ch, k=3, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        if nrm is None:
            nrm = 'identity'
        
        ch1, ch2 = ch

        self.conv1 = conv1_3d((ch1, ch2), k, bias=bias)
        self.norm1 = normlayer3d(ch2, nrm)
        self.act1 = activation(act)

        self.conv2 = conv1_3d((ch2, ch2), k, bias=bias)
        self.norm2 = normlayer3d(ch2, nrm)
        self.act2 = activation(act)

        #self.se = SEBlock(ch2)

        self.skip = skip

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        nn.init.kaiming_uniform_(self.conv2.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)
            self.conv2.weight /= np.sqrt(L)

    def forward(self, x):

        #out = self.se(out)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        return out


class VnetDownBlock(nn.Module):
    def __init__(self, ch, k=2,stride=2, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        ch1, ch2 = ch

        self.conv1 = conv1_3d((ch1, ch2), k,stride=stride, bias=bias)
        self.act1 = activation(act)

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        return out

class VnetUpBlock(nn.Module):
    def __init__(self, ch, k=2,stride=2, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        ch1, ch2 = ch

        self.conv1 = conv1_t_3d((ch1, ch2), k,stride=stride, bias=bias)
        self.act1 = activation(act)

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        return out

class VnetBasicBlock(nn.Module):
    def __init__(self, ch, k=5, act='prelu', skip=False, nrm=None, bias=False,
                 depth=0, L=1, alpha=1):
        super().__init__()

        ch1, ch2 = ch

        self.conv1 = conv1_3d((ch1, ch2), k, bias=bias)
        self.act1 = activation(act)

        # init
        mo = 'fan_out'
        nl = 'leaky_relu' if act == 'prelu' else 'relu'

        nn.init.kaiming_uniform_(self.conv1.weight, mode=mo, nonlinearity=nl)
        with th.no_grad():
            self.conv1.weight /= np.sqrt(L)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        #out += x
        return out


class Bottleneck(nn.Module):

    def __init__(self, ch, k=3, alpha=4, act='prelu', skip=False, nrm=None,
                 bias=False, depth=0, L=1):
        super().__init__()

        if nrm is None:
            nrm = 'identity'

        ch1, ch2 = ch
        ch3 = ch2 // alpha

        if ch1 == ch2:
            self.bottle = nn.Identity()
        else:
            self.bottle = conv1((ch1, ch2), k=1, bias=bias)

        self.conv1 = conv1((ch2, ch3), k, bias=bias)
        self.norm1 = normlayer(ch3, nrm)
        self.act1 = activation(act)

        self.conv2 = conv1((ch3, ch3), k, bias=bias)
        self.norm2 = normlayer(ch3, nrm)
        self.act2 = activation(act)

        self.conv3 = conv1((ch3, ch2), k, bias=bias)
        self.norm3 = normlayer(ch2, nrm)
        self.act3 = activation(act)

        #self.se = SEBlock(ch2)

        self.skip = skip

    def forward(self, x):
        out = self.bottle(x)
        identity = out

        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.norm3(out)
        out = self.act3(out)

        #out = self.se(out)

        if self.skip:
            out += identity

        return out


class SEBlock(nn.Module):

    def __init__(self, ch, ratio=16):
        super(SEBlock, self).__init__()

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(ch, ch // ratio, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(ch // ratio, ch, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        out = self.pool(x).view(b, c)
        out = self.fc1(out)
        out = self.act(out)
        out = self.fc2(out)
        out = self.sig(out)
        return x * out.view(b, c, 1, 1).expand_as(x)


class DenseBlock(nn.Module):

    def __init__(self, ch, nblocks=3, k=3, act='prelu', skip=False, nrm=None,
                 bias=False, depth=0, L=1):
        super(DenseBlock, self).__init__()

        ch1, ch2 = ch

        self.ch = ch
        self.nblocks = nblocks

        if ch1 == ch2:
            self.bottle1 = nn.Identity()
        else:
            self.bottle1 = conv1((ch1, ch2), k=1, bias=bias)


        blocks = nn.ModuleList()
        for i in range(nblocks):
            ch = (2**i*ch2, 2**i*ch2)
            b = BasicBlock(ch, k=k, act=act, skip=skip, nrm=nrm, bias=bias,
                           depth=depth, L=L)
            blocks.append(b)

        self.blocks = blocks
        self.bottle2 = conv1((2**nblocks*ch2, ch2), k=1, bias=bias)
        #self.se = SEBlock(ch2)

    def forward(self, x):
        out = self.bottle1(x)
        identity = out

        for block in self.blocks:
            u = block(out)
            out = th.cat((out, u), dim=1)

        out = self.bottle2(out)
        #out = self.se(out)
        out += identity

        return out


class DenseGroups(nn.Module):

    def __init__(self, ch, ngroups, group=3, nblocks=3, k=3, act='prelu',
                 skip=False, nrm=None, bias=False, depth=0, L=1):
        super(DenseGroups, self).__init__()

        ch1, ch2 = ch

        self.ch = ch
        self.ngroups = ngroups
        self.group = group

        if ch1 == ch2:
            self.bottle1 = nn.Identity()
        else:
            self.bottle1 = conv1((ch1, ch2), k=1, bias=bias)

        blocks = nn.ModuleList()
        bottles = nn.ModuleList()

        for i in range(ngroups):
            for j in range(group):
                b = DenseBlock((ch2, ch2), nblocks=nblocks, k=k, act=act,
                               skip=skip, nrm=nrm, bias=bias, depth=depth, L=L)
                blocks.append(b)

                b = nn.Sequential(*[
                    conv1(((j+2)*ch2, ch2), k=1, bias=bias),
                    conv1((ch2, ch2), k=k, bias=bias)
                ])
                bottles.append(b)

        self.blocks = blocks
        self.bottles = bottles

    def forward(self, x):
        u = self.bottle1(x)
        identity = us = u

        for (i, (block, bottle)) in enumerate(zip(self.blocks, self.bottles)):
            u = block(u)
            us = th.cat((us, u), dim=1)
            u = bottle(us)

            if (i+1) % self.group == 0:
                us = u
                identity = identity + u
                u = identity

        return u


class DenseConv0(nn.Module):

    def __init__(self, ch, k=3, act='prelu', nrm=None, bias=False):
        super(DenseConv, self).__init__()

        if nrm is None:
            nrm = 'identity'

        ch1, ch2 = ch

        self.conv = conv1((ch1, ch2), k, bias=bias)
        self.norm = normlayer(ch2, nrm)
        self.act = activation(act)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        out = self.act(out)
        out = th.cat((x, out), 1)
        return out


class DenseBlock0(nn.Module):

    def __init__(self, ch, G, nconvs, k=3, act='prelu', nrm=None, bias=False,
                 D=0, L=1):
        super(DenseBlock, self).__init__()

        ch1, ch2 = ch

        self.ch = ch
        self.G = G
        self.nconvs = nconvs

        n = self.nconvs

        C = []
        for i in range(n):
            C.append(DenseConv((ch1+i*G, G), k=k, act=act, nrm=nrm, bias=bias))

        self.convs = nn.Sequential(*C)
        self.bottle = conv1((ch1+n*G, ch2), k=1, bias=bias)

    def forward(self, x):
        identity = x
        out = self.convs(x)
        out = self.bottle(out)
        return out + identity


class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * th.sigmoid(x)


def activation(name):
    name = name.lower()

    if name == 'relu':
        #m = nn.ReLU(inplace=True)
        m = nn.ReLU()

    elif name == 'elu':
        m = nn.ELU(inplace=True)

    elif name == 'celu':
        m = nn.CELU(inplace=True)

    elif name == 'selu':
        #m = nn.SELU(inplace=True)
        m = nn.SELU()

    elif name == 'prelu':
        m = nn.PReLU()

    elif name == 'swish':
        m = Swish()
    elif name == 'sigmoid':
        m = nn.Sigmoid()
    else:
        raise ValueError("unknown activation")

    return m


def normlayer(ch, name='identity'):
    name = name.lower()

    if name == 'identity':
        m = nn.Identity()

    elif name == 'batch':
        m = nn.BatchNorm2d(ch)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif name == 'instance':
        m = nn.InstanceNorm2d(ch)

    else:
        raise ValueError("unknown normlayer")

    return m

def normlayer3d(ch, name='identity'):
    name = name.lower()

    if name == 'identity':
        m = nn.Identity()

    elif name == 'batch':
        m = nn.BatchNorm3d(ch)
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

    elif name == 'instance':
        m = nn.InstanceNorm3d(ch)

    else:
        raise ValueError("unknown normlayer")

    return m

