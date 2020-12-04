import torch as th


def add_model_args(parser, args):
    Model = _get_model(args.model)
    return Model.add_model_args(parser)

def create_model(args):
    Model = _get_model(args.model)
    m = Model(**vars(args))

    # if args.ckp is not None:
    #     ckp = th.load(args.ckp)
    #     m.load_state_dict(ckp)

    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print('Number of parameters: {:.3e}'.format(n))

    return m

def _get_model(name):
    name = name.lower()
    if name == 'unet':
        from .unet import UNet as M
    elif name == 'unet3d':
        from .unet3d import UNet3D as M
    elif name == 'dunet':
        from .denseunet import DenseUNet as M
    elif name == 'rdn':
        from .rdn import RDN as M
    elif name == 'vnet':
        from .vnet import VNet as M
    else:
        raise ValueError("unknown model: {}".format(name))

    return M
