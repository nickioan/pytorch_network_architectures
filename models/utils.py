import pathlib
from argparse import ArgumentParser

import torch as th


def average_models(logdir):
    MODEL_NAME = 'best_model_avg.pt'
    files = list(pathlib.Path(logdir).glob('best_model*.pt'))
    files = [file for file in files if not file.match(MODEL_NAME)]
    mfile = pathlib.Path(logdir).joinpath(MODEL_NAME)
    _average_models(files, mfile)


def average_checkpoints(logdir):
    MODEL_NAME = 'best_model_avg.pt'
    files = list(pathlib.Path(logdir).glob('checkpoint*.pt'))
    files = [file for file in files if not file.match(MODEL_NAME)]
    mfile = pathlib.Path(logdir).joinpath(MODEL_NAME)
    _average_models(files, mfile, isckp=True)


def _average_models(infiles, outfile, isckp=False):
    m = th.load(infiles[0])
    m = m if not isckp else m['model']
    for file in infiles[1:]:
        ckp = th.load(file)
        ckp = ckp if not isckp else ckp['model']
        for k in m.keys():
            m[k] += ckp[k]

    N = len(infiles)
    for k in m.keys():
        m[k] /= N

    th.save(m, outfile)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('dir', type=str)
    parser.add_argument('--ckp', default=False, action='store_true')
    args = parser.parse_args()

    if args.ckp:
        average_checkpoints(args.dir)
    else:
        average_models(args.dir)
