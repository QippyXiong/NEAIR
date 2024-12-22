from os import PathLike
from dataclasses import dataclass
from typing import Optional, Literal

import numpy as np

from .utils import datacls_format_str

class IMetrics:

    def __init__(self) -> None:
        for method in ['__gt__', '__lt__', '__str__', 'save', 'load', 'accumulate', \
                       'average', 'asdict', '__float__']:
            assert getattr(self, method) is not getattr(IMetrics, method), \
                f"Method {method} should be implemented in subclass"

    def __gt__(self, metric: 'IMetrics') -> bool:
        raise NotImplementedError("This method __gt__ should be implemented in subclass")
    
    def __lt__(self, metric: 'IMetrics') -> bool:
        return not self > metric

    def __str__(self) -> str:
        raise NotImplementedError("This method __str__ should be implemented in subclass")
    
    def save(self, save_path: PathLike) -> None:
        raise NotImplementedError("This method save should be implemented in subclass")

    @classmethod
    def load(cls, save_path: PathLike) -> 'IMetrics':
        raise NotImplementedError("This method load should be implemented in subclass")
    
    def accumulate(self, *args) -> None:
        raise NotImplementedError("This method accumulate should be implemented in subclass")
    
    def average(self) -> None:
        raise NotImplementedError("This method average should be implemented in subclass")

    def asdict(self) -> dict:
        raise NotImplementedError("This method asdict should be implemented in subclass")
    
    def __float__(self) -> float:
        raise NotImplementedError("This method __float__ should be implemented in subclass")

    def metric_dict(self) -> dict:
        raise NotImplementedError("This method metric_dict should be implemented in subclass")

class IDataset:
    def __init__(self) -> None:
        for method in ['__len__', '__getitem__', 'dump_config', 'from_config']:
            assert getattr(self, method) is not getattr(IDataset, method), \
                f"Method {method} should be implemented in subclass"

    def __len__(self) -> int:
        raise NotImplementedError("This method __len__ should be implemented in subclass")
    
    def __getitem__(self, idx: int) -> np.ndarray:
        raise NotImplementedError("This method __getitem__ should be implemented in subclass")
    
    def dump_config(self) -> dict:
        raise NotImplementedError("This method dump_settings should be implemented in subclass")
    
    @classmethod
    def from_config(cls, config: dict):
        raise NotImplementedError("This method from_settings should be implemented in subclass")


@dataclass(unsafe_hash=True)
class TrainParams:
    # normal hyperparameters
    lr: float
    min_lr: float

    batch_size: int
    # eval_batch_size: int  # set the eval batch size into the control params
                            # * here we set the dropout, but which is ticky is that the dropout is
    weight_decay: float     # * always set during model initialization, but we have to compare it 
    dropout: float          # * as a training parameter

    optimizer_cls: str      # optimizer_cls is used by `getattr(torch.optim, optimizer_cls)`
    lr_scheduler: str       # default usable lr_scheduler is `Linear`
    warmup_proportion: float
    lr_decay: float

    dataset: str            # * `dataset` and `loss_func` attrib both work as identity, they won't
    loss_func: str          # * be used in the deafult `Trainer`

    num_epochs: int 
    num_steps: int

    def __str__(self):
        return datacls_format_str(self)
    
    def __post_init__(self):
        assert self.lr > 0, "Learning rate should be positive"
        assert self.min_lr > 0, "Minimum learning rate should be positive"
        assert self.batch_size > 0, "Batch size should be positive"
        assert self.weight_decay >= 0, "Weight decay should be non-negative"
        assert 0 <= self.dropout < 1, "Dropout should be in [0, 1)"
        assert self.warmup_proportion >= 0., "Warmup proportion should be non-negative"
        assert self.lr_decay > 0., "Learning rate decay should be non-negative"
        assert self.num_epochs > 0 or self.num_steps > 0, "Steps or epochs must have one "\
                                                          "greater than 0."


@dataclass(unsafe_hash=True)
class TrainerStatus:
    dataset: str
    cur_epoch: int   # * step & epoch are in [1, `num_steps`], [1, `num_epochs`] respectively.
    cur_step:   int  # * 

    best_model: str = ''   # store the best checkpoint name
    model_type: str = ''   # string represents a , by default set to ModelCls.__name__

    best_epoch: int = 1  # added at prepare_epoch, begin at 1
    best_step: int  = 1  # added at prepare_step, begin at 1
    training_seconds: int = 0

    def __str__(self):
        return datacls_format_str(self)


@dataclass(unsafe_hash=True)
class ControlParams:
    r"""
    Recording data for training process control.
    """
    loop_control: Literal['epoch','step']      # 'epoch' or 'step'
    valid_span_epoch: int  
    save_span_epoch : int  

    eval_batch_size: int  # move this to ctrl params since we don't want to compare it

    valid_span_steps: int  # * use -1 for unused, which means if model train, save according to 
    save_span_steps : int  # * the number of steps, you should set num_epochs, valid_span_epochs 
                           # * and save_span_epochs to -1, verse visa.
    loss_log_steps:   int 
    loss_log_epochs:  int 
    loss_log_control: Optional[Literal['epoch', 'step']] = None

    # early stop mechanism
    early_stop_prop: Optional[float] = None
    # * Gradient accumulation, Notice the minibatch size is still batch_size.
    # * This would split batch_size B into B/acc_step size microbatches
    grad_acc_step: int = 1
    valid_after_train: bool = False


    def __str__(self):
        return datacls_format_str(self)
    
    def __post_init__(self):
        assert self.grad_acc_step > 0, "Gradient accumulation step should be positive"
        if self.loop_control == 'step':
            assert self.valid_span_steps > 0, "Valid span steps should be positive"
            assert self.save_span_steps > 0, "Save span steps should be positive"
        else:
            assert self.valid_span_epoch > 0, "Valid span epochs should be positive"
            assert self.save_span_epoch > 0, "Save span epochs should be positive"
            
        if self.loss_log_control is None:
            self.loss_log_control = self.loop_control
        
        assert not (self.loss_log_steps <= 0 and self.loss_log_control == 'step'),\
            "Loss log steps should be positive"
        assert not (self.loss_log_epochs <= 0 and self.loss_log_control == 'epoch'),\
            "Loss log epochs should be positive"

@dataclass(unsafe_hash=True)
class Hyperparams:

    def __str__(self):
        return datacls_format_str(self)

@dataclass(unsafe_hash=True)
class DatasetConfig:

    def __str__(self):
        return datacls_format_str(self)
