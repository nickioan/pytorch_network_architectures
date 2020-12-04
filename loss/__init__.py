import torch as th
from .dice import DiceLoss

def create_loss(args):
    name = args.loss.lower()

    if name == 'l1':
        from torch.nn import L1Loss
        loss = L1Loss(reduction='sum')

    elif name == 'l2':
        from torch.nn import MSELoss
        loss = MSELoss(reduction='sum')

    elif name == 'bce':
        from torch.nn import BCELoss
        loss = BCELoss(reduction='sum')
        
    elif name == 'diceloss':
        loss = DiceLoss()
    else:
        raise ValueError('loss must be one of l1, l2, bce,diceloss')

    return loss

# def dice_loss(probs,target):
#     """
#     input is a torch variable of size BatchxnclassesxHxWxD representing log probabilities for each class
#     target is a 1-hot representation of the groundtruth, shoud have same size as the input
#     """
#     eps = 1e-6
#     dims = (2,3,4)

#     intersection = th.sum(probs*target,dims)
#     cardinality = th.sum(probs+target,dims)
#     dice_score = 2. * intersection/(cardinality+eps)
#     return th.mean(1-dice_score)