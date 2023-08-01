
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right
from math import cos, pi

class WarmUpLR(_LRScheduler):
    def __init__(self, lr_scheduler, warmup_steps, eta_min=1e-7):
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.eta_min = eta_min
        super().__init__(lr_scheduler.optimizer, lr_scheduler.last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            return [self.eta_min + (base_lr - self.eta_min) * (self.last_epoch / self.warmup_steps)
                    for base_lr in self.base_lrs]
        return self.lr_scheduler.get_lr()

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        if epoch < self.warmup_steps:
            super().step(epoch)
        else:
            self.last_epoch = epoch
            self.lr_scheduler.step(epoch - self.warmup_steps)

    def state_dict(self):
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_scheduler')}
        state_dict['lr_scheduler'] = self.lr_scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        lr_scheduler = state_dict.pop('lr_scheduler')
        self.__dict__.update(state_dict)
        self.lr_scheduler.load_state_dict(lr_scheduler)


class LRSchedulerWithWarmup(_LRScheduler):
    def __init__(
        self,
        optimizer,
        milestones=(10, 15, 20, 25, 30),
        gamma=0.1,
        mode="cosine",
        warmup_factor=.1,
        warmup_epochs=5,
        warmup_method="linear",
        total_epochs=25,
        target_lr=0,
        power=0.9,
        last_epoch=-1,
    ):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                "Milestones should be a list of"
                " increasing integers. Got {}".format(milestones),
            )
        if mode not in ("step", "exp", "poly", "cosine", "linear"):
            raise ValueError(
                "Only 'step', 'exp', 'poly' or 'cosine' learning rate scheduler accepted"
                "got {}".format(mode)
            )
        if warmup_method not in ("constant", "linear"):
            raise ValueError(
                "Only 'constant' or 'linear' warmup_method accepted"
                "got {}".format(warmup_method)
            )
        self.milestones = milestones
        self.mode = mode
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.total_epochs = total_epochs
        self.target_lr = target_lr
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            elif self.warmup_method == "linear":
                alpha = self.last_epoch / self.warmup_epochs
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
                
            return [base_lr * warmup_factor for base_lr in self.base_lrs]

        if self.mode == "step":
            return [
                base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs
            ]

        epoch_ratio = (self.last_epoch
                        - self.warmup_epochs) / (
            self.total_epochs - self.warmup_epochs
        )

        if self.mode == "exp":
            factor = epoch_ratio
            return [base_lr * self.power ** factor for base_lr in self.base_lrs]
        if self.mode == "linear":
            factor = 1 - epoch_ratio
            return [base_lr * factor for base_lr in self.base_lrs]

        if self.mode == "poly":
            factor = 1 - epoch_ratio
            return [
                self.target_lr + (base_lr - self.target_lr) * self.power ** factor
                for base_lr in self.base_lrs
            ]
        if self.mode == "cosine":
            factor = 0.5 * (1 + cos(pi * epoch_ratio))
            return [
                self.target_lr + (base_lr - self.target_lr) * factor
                for base_lr in self.base_lrs
            ]
        raise NotImplementedError

def build_vanilla_optimizer(cfg, model, trainloader):

    params = []

    print(f'Using {cfg.TRAIN.LR.LR_FACTOR} times learning rate for random init module ')
    
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.TRAIN.LR.BASE_LR
        weight_decay = cfg.TRAIN.WEIGHT_DECAY

        if "cross" in key:
            # use large learning rate for random initialized cross modal module
            lr =  cfg.TRAIN.LR.BASE_LR * cfg.TRAIN.LR.LR_FACTOR # default 5.0
        if "bias" in key:
            lr = cfg.TRAIN.LR.BASE_LR * cfg.TRAIN.LR.BIAS_LR_FACTOR
            weight_decay = cfg.TRAIN.WEIGHT_DECAY_BIAS
        if "classifier" in key or "mlm_head"  in key:
            lr = cfg.TRAIN.LR.BASE_LR * cfg.TRAIN.LR.LR_FACTOR
        if "decoder" in key:
            lr = cfg.TRAIN.LR.BASE_LR * cfg.TRAIN.LR.LR_FACTOR * 2

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        

    optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.TRAIN.LR.BASE_LR)
    
    #step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT*cfg.TRAIN.LR.DELAY , gamma=0.1)
    #scheduler = WarmUpLR(lr_scheduler=step_scheduler, warmup_steps=int(1.*cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))

    scheduler = LRSchedulerWithWarmup(optimizer, 
                                      mode=cfg.TRAIN.LR.MODE, 
                                      warmup_epochs=cfg.TRAIN.LR.WARMUP_EPOCH, 
                                      total_epochs=cfg.TRAIN.EPOCH,
                                      warmup_factor=cfg.TRAIN.LR.WARMUP_FACTOR,
                                      gamma=cfg.TRAIN.LR.GAMMA)
    return optimizer, scheduler

def build_vanilla_optimizer_combine(cfg, params_group, trainloader):
    optimizer = torch.optim.AdamW(params=params_group, lr = cfg.TRAIN.LR.BASE_LR)
    step_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=len(trainloader)*cfg.TRAIN.ONE_EPOCH_REPEAT*cfg.TRAIN.LR.DELAY , gamma=0.1)
    scheduler = WarmUpLR(lr_scheduler=step_scheduler, warmup_steps=int(1.*cfg.TRAIN.LR.WARMUP_EPOCH*len(trainloader)))
    return optimizer, scheduler


def freeze_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_params(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def freeze_bn(model: nn.Module) -> None:
    def set_bn_eval(m) -> None:
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.eval()
    model.apply(set_bn_eval)


def unfreeze_bn(model: nn.Module) -> None:
    def set_bn_train(m) -> None:
        classname = m.__class__.__name__
        if classname.find('BatchNorm') != -1:
            m.train()
    model.apply(set_bn_train)


class FreezeBackbone(object):
    def __init__(self, model: nn.Module, freeze_epoch=0):
        super().__init__()
        self.model = model
        self.freeze_epoch = freeze_epoch
        self.backbone_name = ['vis_backbone', 'text_backbone']

    def start_freeze_backbone(self):
        if self.freeze_epoch <= 0:
            return
        for name in self.backbone_name:
            layer = self.model.module._modules[name]
            freeze_params(layer)
            freeze_bn(layer)
            print(f'====> Freeze {name}')

    def on_train_epoch_start(self, epoch) -> None:
        if self.freeze_epoch <= 0:
            return
        if epoch == self.freeze_epoch:
            for name in self.backbone_name:
                layer = self.model.module._modules[name]
                unfreeze_params(layer)
                unfreeze_bn(layer)
                print(f'====> Unfreeze {name}')