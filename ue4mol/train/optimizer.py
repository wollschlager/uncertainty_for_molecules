from torch.optim.lr_scheduler import *
from torch.optim import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch._six import inf

class EarlyStopping:
    def __init__(self, patience, threshold=1e-4, threshold_mode='rel', verbose=True, delta=0, mode='min', **kwargs) -> None:
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self._init_is_better(mode, threshold, threshold_mode)
        self.best = self.mode_worse
        
           
    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
    
    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold
        
    def __call__(self, current):
        
        if self.is_better(current, self.best):
            if self.verbose:
                print('new best metric: ', current, ' setting counter back to zero')
            self.best = current
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            
class Optimizer:
    def __init__(
        self, 
        optimizer_type: str, 
        optim_args: dict, 
        schedule_type: str=None, 
        sched_args: dict=None
    ) -> None:

        self.optim = optimizer_map[optimizer_type]
        self.optim_args = optim_args
        self.schedule = False
        self.schedule_type = schedule_type
        if schedule_type is not None and sched_args is not None:
            self.sched = scheduler_map[schedule_type]
            self.sched_args = sched_args
            self.schedule = True

    def initialize(self, model_params):
        scheduler = None
        optimizer = self.optim(model_params, **self.optim_args)
        if self.schedule:
            scheduler = self.sched(optimizer, **self.sched_args)
            if self.schedule_type == 'reduce_on_plateau':
                scheduler = {"scheduler": scheduler, "monitor": "train_loss"}
            else:
                scheduler = {"scheduler": scheduler, "frequency": 1, "interval": "step"}
        return optimizer, scheduler


class LinearWarmupExponentialDecay(LambdaLR):
    """This schedule combines a linear warmup with an exponential decay.

    Parameters
    ----------
        optimizer: Optimizer
            Optimizer instance.
        decay_steps: float
            Number of steps until learning rate reaches learning_rate*decay_rate
        decay_rate: float
            Decay rate.
        warmup_steps: int
            Total number of warmup steps of the learning rate schedule.
        staircase: bool
            If True use staircase decay and not (continous) exponential decay.
        last_step: int
            Only needed when resuming training to resume learning rate schedule at this step.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps,
        decay_steps,
        decay_rate,
        staircase=False,
        last_step=-1,
        verbose=False,
    ):
        assert decay_rate <= 1

        if warmup_steps == 0:
            warmup_steps = 1

        def lr_lambda(step):
            # step starts at 0
            warmup = min(1 / warmup_steps + 1 / warmup_steps * step, 1)
            exponent = step / decay_steps
            if staircase:
                exponent = int(exponent)
            decay = decay_rate ** exponent
            return warmup * decay

        super().__init__(optimizer, lr_lambda, last_epoch=last_step, verbose=verbose)

scheduler_map = {
    'lambda_lr': LambdaLR,
    'step_lr': StepLR,
    'linear_warmup_exponential_decay': LinearWarmupExponentialDecay,
    'reduce_on_plateau': ReduceLROnPlateau
}

optimizer_map = {
    'sgd': SGD,
    'adam': Adam,
    'adamW': AdamW,
}