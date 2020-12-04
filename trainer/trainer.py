import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import wandb

from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, EarlyStopping
from ignite.handlers import TerminateOnNan, global_step_from_engine
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.time_profilers import BasicTimeProfiler
from ignite.utils import setup_logger

from .utils import print_model_summary
from optim import create_lr_scheduler
from utils.metrics import accuracy
from utils.metrics import accuracy_segmentation


# Events in order
#   Events.STARTED
#   Events.EPOCH_STARTED
#   Events.GET_BATCH_STARTED
#   Events.DATALOADER_STOP_ITERATION
#   Events.GET_BATCH_COMPLETED
#   Events.ITERATION_STARTED
#   Events.ITERATION_COMPLETED
#   Events.TERMINATE_SINGLE_EPOCH
#   Events.TERMINATE
#   Events.EPOCH_COMPLETED
#   Events.COMPLETED
#   Events.EXCEPTION_RAISED


def create_trainer(loader, model, opt, loss_fn, device, args):

    def _update(engine, batch):
        model.train()

        x = batch['x'].to(engine.state.device, non_blocking=True)
        y = batch['y'].to(engine.state.device, non_blocking=True)
        m = batch['m'].to(engine.state.device, non_blocking=True)
        opt.zero_grad()
        y_pred = model(x)

        softmax = nn.Softmax()
        masked_loss = softmax(y_pred)
        #masked_loss = y_pred*m
        loss = loss_fn(masked_loss, y)
        if m.sum().item() / m.numel() > 0.7:
            loss.backward()
            opt.step()
        masked_loss = (masked_loss>0.5).float()
        acc = accuracy_segmentation(masked_loss[:,1,:,:,:],y[:,1,:,:,:])

        return {
            'x': x.detach(),
            'y': y.detach(),
            'm': m.detach(),
            'y_pred': y_pred.detach(),
            'loss': loss.item(),
            'acc' : acc
        }

    def _inference(engine, batch):
        model.eval()

        with th.no_grad():
            x = batch['x'].to(engine.state.device, non_blocking=True)
            y = batch['y'].to(engine.state.device, non_blocking=True)
            m = batch['m'].to(engine.state.device, non_blocking=True)

            y_pred = model(x)
            
            softmax = nn.Softmax(dim=1)
            masked_loss = softmax(y_pred)
            #masked_loss = y_pred*m
            loss = loss_fn(masked_loss, y)
            masked_loss = (masked_loss[-3:]>0.5).float()
            acc = accuracy_segmentation(masked_loss[:,1,:,:,:],y[:,1,:,:,:])

        return {
            'x': x.detach(),
            'y': y.detach(),
            'm': m.detach(),
            'y_pred': y_pred.detach(),
            'loss': loss.item(),
            'acc' : acc
        }


    #wandb.watch(model, log ='all')

    trainer = Engine(_update)
    evaluator = Engine(_inference)

    profiler = BasicTimeProfiler()
    profiler.attach(trainer)
    logdir = args.logdir
    save_ = (not args.devrun) and (not args.nosave)

    # initialize trainer state
    trainer.state.device = device
    trainer.state.hparams = args
    trainer.state.save = save_
    trainer.state.logdir = logdir

    trainer.state.df = defaultdict(dict)
    trainer.state.metrics = dict()
    trainer.state.val_metrics = dict()
    trainer.state.best_metrics = defaultdict(list)
    trainer.state.gradnorm = defaultdict(dict)

    # initialize evaluator state
    evaluator.logger = setup_logger('evaluator')
    evaluator.state.device = device
    evaluator.state.df = defaultdict(dict)
    evaluator.state.metrics = dict()

    pbar = ProgressBar(persist=True)
    ebar = ProgressBar(persist=False)

    pbar.attach(trainer, ['loss'])
    ebar.attach(evaluator, ['loss'])

    pbar.attach(trainer,['acc'])
    ebar.attach(evaluator,['acc'])

    # model summary
    if args.model_summary:
        trainer.add_event_handler(
            Events.STARTED,
            print_model_summary, model
        )

    # terminate on nan
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        TerminateOnNan(lambda x: x['loss'])
    )

    # metrics
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        _metrics
    )

    evaluator.add_event_handler(
        Events.ITERATION_COMPLETED,
        _metrics
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        _metrics_mean
    )

    evaluator.add_event_handler(
        Events.COMPLETED,
        _metrics_mean
    )

    trainer.add_event_handler(
        #Events.STARTED | Events.EPOCH_COMPLETED,
        Events.EPOCH_COMPLETED,
        _evaluate, evaluator, loader
    )

    # logging
    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        _log_metrics
    )

    # early stopping
    if args.early_stopping > 0:
        es_p = args.early_stopping
        es_s = lambda engine: -engine.state.metrics['loss']
        evaluator.add_event_handler(
            Events.COMPLETED,
            EarlyStopping(patience=es_p, score_function=es_s, trainer=trainer)
        )

    # lr schedulers
    if args.epoch_length is None:
        el = len(loader['train'])
    else:
        el = args.epoch_length

    if args.lr_scheduler is not None:
        lr_sched = create_lr_scheduler(opt, args, num_steps=el)

        if args.lr_scheduler != 'plateau':
            def _sched_fun(engine):
                lr_sched.step()
        else:
            def _sched_fun(engine):
                e = engine.state.epoch
                v = engine.state.val_metrics[e]['nmse']
                lr_sched.step(v)

        if args.lr_scheduler == 'linearcycle':
            trainer.add_event_handler(Events.ITERATION_STARTED, lr_sched)
        else:
            trainer.add_event_handler(Events.EPOCH_COMPLETED, _sched_fun)

    # FIXME: warmup is modifying opt base_lr -> must create last
    if args.lr_warmup > 0:
        wsched = create_lr_scheduler(opt, args, 'warmup', num_steps=el)
        wsts = wsched.total_steps
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED(event_filter=lambda _, i: i <= wsts),
            lambda _: wsched.step()
        )

    # saving
    if save_:
        to_save = {
            'model': model,
            'optimizer': opt,
            'trainer': trainer,
            'evaluator': evaluator
        }

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            Checkpoint(to_save, DiskSaver(logdir), n_saved=3)
        )

        # handler = Checkpoint(
        #     {'model': model},
        #     DiskSaver(logdir),
        #     n_saved = 3,
        #     filename_prefix = 'best',
        #     score_function = lambda engine: -engine.state.metrics['nmae'],
        #     score_name = 'val_nmae',
        # )

        # evaluator.add_event_handler(
        #     Events.COMPLETED,
        #     handler
        # )

        # handler = Checkpoint(
        #     {'model': model},
        #     DiskSaver(logdir),
        #     n_saved = 3,
        #     filename_prefix = 'best',
        #     score_function = lambda engine: -engine.state.metrics['nmse'],
        #     score_name = 'val_nmse',
        # )

        # evaluator.add_event_handler(
        #     Events.COMPLETED,
        #     handler
        # )

        # handler = Checkpoint(
        #     {'model': model},
        #     DiskSaver(logdir),
        #     n_saved = 3,
        #     filename_prefix = 'best',
        #     score_function = lambda engine: engine.state.metrics['R2'],
        #     score_name = 'val_R2',
        # )

        # evaluator.add_event_handler(
        #     Events.COMPLETED,
        #     handler
        # )

        trainer.add_event_handler(
            Events.EPOCH_COMPLETED,
            _save_metrics
        )

        # timer
        trainer.add_event_handler(
            Events.COMPLETED | Events.TERMINATE,
            lambda _: profiler.write_results(logdir + '/time.csv')
        )

    return trainer


def _metrics(engine: Engine) -> None:
    i = engine.state.iteration
    e = engine.state.epoch

    l = engine.state.output['loss']
    x = engine.state.output['y_pred']
    y = engine.state.output['y']
    m = engine.state.output['m']
    a = engine.state.output['acc']
    #a = accuracy(m*x, y)
    

    engine.state.df[i]['epoch'] = e
    engine.state.df[i]['loss'] = l
    engine.state.metrics['loss'] = l
    engine.state.df[i]['acc'] = a
    engine.state.metrics['acc'] = a
    # for k, v in a.items():
    #     engine.state.df[i][k] = v.item()
    #     engine.state.metrics[k] = v.item()


def _metrics_mean(engine: Engine) -> None:
    e = engine.state.epoch
    df = pd.DataFrame.from_dict(engine.state.df, orient='index')
    df = df[df['epoch'] == e]
    df = df.loc[:, df.columns != 'epoch'].mean()
    d = df.to_dict()
    for k, v in d.items():
        engine.state.metrics[k] = v


def _evaluate(trainer: Engine, evaluator: Engine, loader) -> None:
    e = trainer.state.epoch

    evaluator.run(loader['val'])
    trainer.state.val_metrics[e] = evaluator.state.metrics

    # FIXME
    for k, v in evaluator.state.metrics.items():
        bm = trainer.state.best_metrics['val_' + k]
        bm.append((v, e))
        bm.sort()
        trainer.state.best_metrics['val_' + k] = bm[:3]


def _log_metrics(engine: Engine) -> None:
    e = engine.state.epoch

    tm = engine.state.metrics
    em = dict()
    for k, v in engine.state.val_metrics[e].items():
        em['val_' + k] = v

    logger = logging.getLogger(__name__)
    for k, v in tm.items():
        logger.info('train_' + '{:<5} : {}'.format(k, v))
    for k, v in em.items():
        logger.info('{:<11} : {}'.format(k, v))

    if engine.state.save:
        wandb.log({**tm, **em})


def _save_metrics(engine: Engine):
    df = pd.DataFrame.from_dict(engine.state.df, orient='index')
    df.reset_index(inplace=True)
    df.to_feather(engine.state.logdir + '/df.feather')

    df = pd.DataFrame.from_dict(engine.state.val_metrics, orient='index')
    df.reset_index(inplace=True)
    df.to_feather(engine.state.logdir + '/val.feather')



"""
_update and _inference for the CosmosData Dataset (different project)
"""
#def create_trainer(loader, model, opt, loss_fn, device, args):
#
#    def _update(engine, batch):
#        model.train()
#
#        fl = batch[0].unsqueeze_(1).to(engine.state.device, non_blocking=True)
#        cosmos = batch[1].unsqueeze_(1).to(engine.state.device, non_blocking=True)
#        mask = batch[2].unsqueeze_(1).to(engine.state.device, non_blocking=True)
#        D = batch[3].unsqueeze_(1).to(engine.state.device, non_blocking=True)
#        D = D.unsqueeze_(5)
#
#        x0 = fl.clone()
#
#        with th.no_grad():
#            fl *= mask
#            cosmos *= mask
#            fl = th.rfft(fl, 3, onesided=False, normalized=True)
#            x0 = th.irfft(D*fl, 3, onesided=False, normalized=True,
#                          signal_sizes=x0.size()[-3:])
#
#        opt.zero_grad()
#        x = mask*model(x0, fl, D)
#        loss = loss_fn(x, cosmos)
#        loss.backward()
#        opt.step()
#
#        return {
#            'y': cosmos.detach(),
#            'y_pred': x.detach(),
#            'loss': loss.item()
#        }
#
#    def _inference(engine, batch):
#        model.eval()
#
#        with th.no_grad():
#            fl = batch[0].unsqueeze_(1).to(engine.state.device, non_blocking=False)
#            cosmos = batch[1].unsqueeze_(1).to(engine.state.device, non_blocking=False)
#            mask = batch[2].unsqueeze_(1).to(engine.state.device, non_blocking=False)
#            D = batch[3].unsqueeze_(1).to(engine.state.device, non_blocking=False)
#            D = D.unsqueeze_(5)
#
#            fl *= mask
#            cosmos *= mask
#
#            x0 = fl.clone()
#            fl = th.rfft(fl, 3, onesided=False, normalized=True)
#            x0 = th.irfft(D*fl, 3, onesided=False, normalized=True,
#                          signal_sizes=x0.size()[-3:])
#
#            x = model(x0, fl, D)
#            x *= mask
#            loss = loss_fn(x, cosmos)
#
#        return {
#            'y': cosmos.detach(),
#            'y_pred': x.detach(),
#            'loss': loss.item()
#        }
