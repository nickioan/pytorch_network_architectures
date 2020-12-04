import logging
from argparse import ArgumentParser

import torch as th
import torch.backends.cudnn as cudnn

import wandb

from data import create_dataloader
from loss import create_loss
from models import add_model_args, create_model
from optim import create_optimiser
from trainer import create_trainer
from utils import init_logging, seed_all


def main(args):

    logdir = init_logging(args)
    logger = logging.getLogger(__name__)

    args.logdir = logdir

    if args.cpu or not th.cuda.is_available():
        device = th.device('cpu')
    else:
        device = th.device('cuda')
        cudnn.enabled = True
        cudnn.benchmark = True

    if not args.devrun and not args.nosave:
        wandb.init(config=args, dir=logdir, project=args.project)

        if args.name is not None:
            wandb.run.name = args.name
        # else:
        #     wandb.run.name = wandb.run.id

    seed_all(args.seed)

    logger.info('Creating dataloader')
    loader = create_dataloader(args)

    logger.info('Creating model')
    model = create_model(args).to(device)

    logger.info('Creating optimiser')
    opt = create_optimiser(model.parameters(), args)

    logger.info('Creating loss')
    loss = create_loss(args)

    logger.info('Creating trainer')
    trainer = create_trainer(loader, model, opt, loss, device, args)

    epochs = args.epochs
    epoch_length = args.epoch_length

    logger.info('Starting trainer')
    wandb.watch(model,log="all",log_freq=1)
    trainer.run(loader['train'], max_epochs=epochs, epoch_length=epoch_length)


if __name__ == '__main__':
    # select gpu with CUDA_VISIBLE_DEVICES=0 python train.py
    parser = ArgumentParser(add_help=False)

    # model
    parser.add_argument(
        '--model',
        default='unet',
        choices = ['unet', 'dunet', 'rdn','unet3d','vnet'],
        type = str,
    )

    parser.add_argument(
        '--seed',
        default = 733,
        type = int
    )

    parser.add_argument(
        '--project',
        default = 'playground',
        type = str
    )

    parser.add_argument(
        '--name',
        default = None,
        type = str
    )

    parser.add_argument(
        '--cpu',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--ckp',
        default = None,
        type = str
    )

    parser.add_argument(
        '--suffix',
        default = None,
        type = str
    )

    # directories
    parser.add_argument(
        '--logdir',
        default = '/srv/data/coopar6/microbleeds_ai/log/paper',
        type = str
    )

    parser.add_argument(
        '--datadir',
        default = '/srv/data/coopar6/microbleeds_ai/MicrobleedData',
        type = str
    )

    # training
    parser.add_argument(
        '--epochs',
        default = 1000,
        type = int
    )

    parser.add_argument(
        '--epoch_length',
        default = None,
        type = int
    )

    parser.add_argument(
        '--batch_size',
         default = 32,
         type = int
    )

    parser.add_argument(
        '--padding',
        default = 'zeros',
        choices = ['zeros', 'reflect', 'replicate'],
        type = str
    )

    parser.add_argument(
        '--patch',
        default = [96,96,96],
        nargs = '+',
        type = int
    )

    parser.add_argument(
        '--no_augment',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--num_workers',
        default = 8,
        type = int
    )

    # loss
    parser.add_argument(
        '--loss',
        default = 'diceloss',
        choices = ['l1', 'l2','bce','diceloss'],
        type = str
    )

    # optim
    parser.add_argument(
        '--optim',
        default = 'adam',
        choices = ['adagrad', 'adam', 'rmsprop', 'sgd'],
        type = str
    )

    parser.add_argument(
        '--lr',
        default = 1e-4,
        type = float
    )

    # optim.adagrad
    parser.add_argument(
        '--lr_decay',
        default = 0,
        type = float
    )

    # optim.adam
    parser.add_argument(
        '--beta1',
        default = 0.9,
        type = float
    )

    parser.add_argument(
        '--beta2',
        default = 0.999,
        type = float
    )

    # optim.rmsprop
    parser.add_argument(
        '--alpha',
        default = 0.99,
        type = float
    )

    # optim.common
    parser.add_argument(
        '--momentum',
        default = 0.9,
        type = float
    )

    parser.add_argument(
        '--weight_decay',
        default = 0,
        type = float
    )

    # optim.lr_scheduler
    parser.add_argument(
        '--lr_scheduler',
        default = None,
        choices = ['step', 'multistep', 'exponential', 'plateau', 'linearcycle'],
        type = str
    )

    parser.add_argument(
        '--lr_warmup',
        default = 0,
        type = int
    )

    parser.add_argument(
        '--lr_start',
        default = 1e-8,
        type = float
    )

    # optim.lr_scheduler.StepLR
    parser.add_argument(
        '--step_size',
        default = 30,
        type = int
    )

    # optim.lr_scheduler.MultiStepLR
    parser.add_argument(
        '--milestones',
        default = [30, 60, 90],
        nargs = '+',
        type = int
    )

    # optim.lr_scheduler.ReduceLROnPlateau
    parser.add_argument(
        '--patience',
        default = 8,
        type = int
    )

    parser.add_argument(
        '--patience_factor',
        default = 2,
        type = int
    )

    parser.add_argument(
        '--max_patience',
        default = 64,
        type = int
    )

    parser.add_argument(
        '--min_lr',
        default = 1e-6,
        type = float
    )

    parser.add_argument(
        '--threshold',
        default = 1e-4,
        type = float
    )

    # optim.lr_scheduler.common
    parser.add_argument(
        '--gamma',
        default = 0.25,
        type = float
    )

    parser.add_argument(
        '--early_stopping',
        default = 128,
        type = int
    )

    parser.add_argument(
        '--model_summary',
        default = False,
        action = 'store_true',
    )

    # development
    parser.add_argument(
        '--devrun',
        default = False,
        action = 'store_true'
    )

    parser.add_argument(
        '--nosave',
        default = False,
        action = 'store_true'
    )

    tmp_args, _ = parser.parse_known_args()
    parser = add_model_args(parser, tmp_args)

    args = parser.parse_args()

    main(args)
