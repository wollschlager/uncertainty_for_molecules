from .optimizer import Optimizer
from ..models.interfaces import LightningInterface
import pytorch_lightning as pl
import torch
from torch_ema import ExponentialMovingAverage
from typing import Optional, Callable
from copy import deepcopy


class ModelWrapper(pl.LightningModule):
    def __init__(
        self,
        model: LightningInterface,
        optimizer: Optimizer,
        train_loader,
        val_loader,
        ema_decay,
        ood_loader=None,
        batch_size: int=52,
        unfreeze_after: int=None,
        save_ema=False,
    ):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.epoch = 0
        self.idx = 0
        self.ood_loader = ood_loader
        self.ema = ExponentialMovingAverage(self.model.cuda().parameters(), decay=ema_decay)
        self.unfreeze_after = unfreeze_after
        #self.save_hyperparameters()
        self.optimizer_steps = 0
        self.training_steps = 0
        self.save_ema = save_ema
    
    def log_metrics(self, dataset_name, metrics, batch_size):
        for key in metrics:
            self.log(dataset_name+'_'+key, metrics[key], batch_size=batch_size)
        
    def configure_optimizers(self):
        optim, sched = self.optimizer.initialize(self.model.parameters())
        if sched is None: 
            return [optim]
        return [optim], [sched]
    
    def optimizer_step(
            self,
            epoch: int,
            batch_idx: int,
            optimizer: Optimizer,
            optimizer_idx: int,
            second_order_closure: Optional[Callable] = None,
            on_tpu: bool = False,
            using_native_amp: bool = False,
            using_lbfgs: bool = False,
    ) -> None:
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, second_order_closure)
        if self.ema is not None:
            self.ema.update(self.model.parameters())
        self.optimizer_steps += 1

    def training_step(self, batch, batch_idx):
        current_bs = max(batch.batch)+1
        self.optimizers().zero_grad()
        loss, metrics = self.model.execution_step(batch)
        self.log("train_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True, batch_size=current_bs)
        self.log("learning_rate", self.optimizers().param_groups[0]['lr'])
        self.log_metrics("train", metrics, batch_size=current_bs)
        self.training_steps += 1
        self.log("training_steps", self.training_steps, batch_size=current_bs)
        self.log("optimizer_steps", self.optimizer_steps, batch_size=current_bs)
        return loss

    def validation_step(self, batch, batch_idx):
        current_bs = max(batch.batch)+1
        with torch.no_grad():
            with self.ema.average_parameters():
                loss, metrics = self.model.execution_step(batch)
                self.log("val_loss", loss.item(), on_epoch=True, on_step=False, batch_size=current_bs)
                self.log_metrics("val", metrics, batch_size=current_bs)

    def test_step(self, batch, batch_idx):
        current_bs = max(batch.batch)+1
        with torch.no_grad():
            with self.ema.average_parameters():
                loss, metrics = self.model.execution_step(batch)
        self.log("ood_loss", loss.item(), on_epoch=True, on_step=False, batch_size=current_bs)
        self.log_metrics("ood", metrics, batch_size=current_bs)
                    
    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self): #ood dataloader
        if self.ood_loader is not None:
            return self.ood_loader
        else:
            return None

    def on_save_checkpoint(self, checkpoint):
        checkpoint['state_dict'] = self.state_dict()
        if self.save_ema:
            with self.ema.average_parameters():
                checkpoint['ema_state_dict'] = deepcopy(self.state_dict())
            
    def on_train_epoch_start(self) -> None:
        self.model.train()

    def on_train_epoch_end(self) -> None:
        if self.ood_loader is not None:
            with torch.no_grad():
                self.evaluate_ood()
        try:
            print(f"Results - Epoch: {self.trainer.current_epoch} - "
                f"ELBO: {self.trainer.logged_metrics['train_loss_epoch']:.2f} - "
                f"Val Likelihood: {self.trainer.logged_metrics['val_likelihood']:.2f} - "
                f"OOD Likelihood: {self.trainer.logged_metrics['ood_likelihood']:.2f}")
        except:
            pass    
        if self.unfreeze_after:
            if self.unfreeze_after == self.current_epoch:
                self.model.model.fe.unfreeze()
        return super().on_train_epoch_end()