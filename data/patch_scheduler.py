import logging

from utils.scheduler import PlateauScheduler


# FIXME: this is a mess
class CyclePatchOnPlateau(PlateauScheduler):

    def __init__(self, patches, verbose=True, **kwargs):
        if not isinstance(patches, list) and not isinstance(patches, tuple):
            patches = [patches]

        self.patches = patches
        self.verbose = verbose
        self.patch = patches[0]
        self.idx = 0
        super().__init__(**kwargs)
        self._patience = self.patience
        self._reset()

    def _reset(self):
        self.idx = 0
        self.patch = self.patches[0]
        super()._reset()

    def step(self, metrics):
        super().step(metrics)
        self._last_patch = self.patches[self.idx]
        return self.idx

    def _fn(self, epoch):
        new_idx = (self.idx + 1) % len(self.patches)
        new_patch = self.patches[new_idx]

        self.idx = new_idx
        self.patch = new_patch

        if self.verbose:
            logger = logging.getLogger(__name__)
            logger.info('Patch scheduler: new patch is {:3d}.'.format(new_patch))

        if self.patience_factor > 0:
            old_p = self.patience
            if new_idx == 0:
                new_p = min(self.patience_factor * self._patience, self.max_patience)
                self._patience = new_p
            else:
                new_p = min(max(old_p // self.patience_factor, 1), self.max_patience)

            self.patience = new_p

            if self.verbose and old_p != new_p:
                logger = logging.getLogger(__name__)
                logger.info('Patch scheduler: new patience is {:3d}'.format(new_p))
