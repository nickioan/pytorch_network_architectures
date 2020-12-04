import logging

from torch.nn import Module
from utils.scheduler import PlateauScheduler


class CycleLossOnPlateau(Module, PlateauScheduler):

    def __init__(self, losses, verbose=True, **kwargs):
        if not isinstance(losses, list) and not isinstance(losses, tuple):
            losses = [losses]

        self.losses = losses
        self.verbose = verbose
        self.idx = 0
        Module.__init__(self)
        PlateauScheduler.__init__(self, **kwargs)
        self._reset()

    def _reset(self):
        self.idx = 0
        self.loss = self.losses[0]
        super()._reset()

    def forward(self, input, target):
        return self.loss(input, target)

    def step(self, metrics):
        ret = super().step(metrics)
        self._last_loss = self.losses[self.idx]
        return self.idx

    def _fn(self, epoch):
        new_idx = (self.idx + 1) % len(self.losses)
        new_loss = self.losses[new_idx]

        self.idx = new_idx
        self.loss = new_loss

        if self.verbose:
            logger = logging.getLogger(__name__)
            logger.info('Loss scheduler: new loss is {}.'.format(new_loss))

        if self.patience_factor > 0:
            old_p = self.patience
            if new_idx == 0:
                new_p = min(self.patience * self.patience_factor, self.max_patience)
                self.patience = new_p
                if self.verbose and old_p != new_p:
                    logger = logging.getLogger(__name__)
                    logger.info('Loss scheduler: new patience is {:3d}'.format(new_p))

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'losses'}
