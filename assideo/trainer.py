from copy import deepcopy
from pathlib import Path

import torch
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .dataset import collate_fn
from .loss import CosFaceLoss, ArcFaceLoss, CustomSoftmax, SubcenterArcFaceLoss


class BaseTrainer:
    def __init__(
            self,
            cfg,
            model,
            data,  # Dataset or DataLoader
            epochs=None,
            criterion=None,
            optimizer=None,
            scheduler=None):
        self.cfg = cfg
        self.model = model
        self.loader = data if isinstance(data, DataLoader) else DataLoader(
            dataset=data,
            batch_size=cfg.batch_size,
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True)
        self.epochs = epochs or cfg.epochs

        self.criterion = self.choose_criterion(criterion or cfg.criterion)
        self.optimizer = self.choose_optimizer(optimizer or cfg.optimizer)
        self.scheduler = self.choose_scheduler(scheduler
                                               or cfg.get('scheduler'))

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        if cfg.force_cpu:
            self.device = 'cpu'

        self.model.to(self.device)
        self.criterion.to(self.device)
        self.stats_update_period = cfg.stats_update_period
        wandb.init(**cfg.wandb, config=cfg)

    def set_description(self, loader, epoch, loss=None):
        loss = f'{loss:.5g}' if loss else '-'
        loader.set_description(f'Epoch: {epoch} | loss: {loss}')

    def train(self, save=True):
        for epoch in range(1, self.epochs + 1):
            running_loss = 0
            loader = tqdm(self.loader)
            self.set_description(loader, epoch)
            for i, data in enumerate(loader):
                imgs, cat_ids = torch.stack(data['image']), torch.tensor(
                    data['category_id'])
                imgs, cat_ids = imgs.to(self.device), cat_ids.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, cat_ids)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if not (i + 1) % self.stats_update_period:
                    self.set_description(loader, epoch, running_loss / (i + 1))
            try:
                latest_lr = self.scheduler.get_last_lr()[0]
            except Exception:
                latest_lr = self.optimizer.param_groups[0]['lr']
            wandb.log({
                'training_loss': running_loss / (i + 1),
                'learning_rate': latest_lr
            })
            running_loss = 0
            if self.scheduler:
                try:
                    self.scheduler.step()
                except TypeError:
                    self.scheduler.step(running_loss)

        if save:
            self.save_state()

    def save_state(self):
        path = Path(self.cfg.saved_model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_type = str(self.cfg.save_type).lower().strip()
        if save_type in ['parameters', 'weights']:
            torch.save(self.model.state_dict(), self.cfg.saved_model_path)
        elif save_type == 'model':
            torch.save(self.model, self.cfg.saved_model_path)
        else:
            raise ValueError(f"Saving type {save_type} does not exist.")

    def choose_criterion(self, criterion):
        if callable(criterion):
            return criterion
        args = deepcopy(criterion)
        name = str(args.pop('name')).lower().strip()
        if name == 'cosface':
            criterion_fn = CosFaceLoss(**args)
        elif name == 'arcface':
            criterion_fn = ArcFaceLoss(**args)
        elif name == 'softmax':
            criterion_fn = CustomSoftmax(**args)
        elif name == 'subcenter_arcface':
            criterion_fn = SubcenterArcFaceLoss(**args)
        else:
            raise ValueError(f"Criterion '{name}' does not exist.")
        return criterion_fn

    def choose_optimizer(self, optimizer):
        if callable(optimizer):
            return optimizer
        args = deepcopy(optimizer)
        name = str(args.pop('name')).lower().strip()
        if name == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), **args)
        elif name == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), **args)
        elif name == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), **args)
        else:
            raise ValueError(f"Optimizer '{name}' does not exist.")
        return optimizer

    def choose_scheduler(self, scheduler):
        if not scheduler:
            return None
        elif callable(scheduler):
            return scheduler
        args = deepcopy(scheduler)
        name = str(args.pop('name')).lower().strip()
        if name == 'reducelronplateau' or name == 'reduceonplateau':
            scheduler = ReduceLROnPlateau(self.optimizer, **args)
        elif name == 'exponentiallr' or name == 'exponential':
            scheduler = ExponentialLR(self.optimizer, **args)
        else:
            raise ValueError(f"Scheduler '{name}' does not exist.")
        return scheduler
