import json
from os import PathLike
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Sequence, Literal, Any
import typing as ty
from datetime import datetime, timedelta
import logging

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from .interfaces import IDataset, IMetrics, TrainerStatus, TrainParams, ControlParams, \
                        Hyperparams, DatasetConfig
from .utils import json_load_dict, json_dump_dict, json_load_dataclass, \
                   json_dump_dataclass


STATUS_FILE_NAME            = 'status.json'
DATASET_FILE_NAME           = 'dataset.json'
HYPERPARAMS_FILE_NAME       = 'hyperparams.json'
TRAIN_PARAMS_FILE_NAME      = 'train_params.json'
CONTROL_PARAMS_FILE_NAME    = 'control_params.json'
TEST_METRICS_FILE_NAME      = 'test_metrics.json'
BEST_VALID_METRICS_FILE_NAME= 'best_valid_metrics.json'
TRAIN_LOG_FILE_NAME         = 'train.log'

MODEL_CKPT_FILE             = 'pytorch_model.bin'
OPTIMIZER_CKPT_FILE         = 'optimizer.bin'

class Trainer:
    r"""
    checkpoint structure:
    root_dir
    - specific_name (self.save_dir, newed by `self.get_save_dir()`)
        - models
            - epoch_n / steps_n
            - ...
        - validations
            - epoch_n / steps_n
            - ...
        - status.json
        - dataset.json
        - hyperparams.json
        - train_params.json
        - control_params.json
        - best_valid_metrics.json
        - test_metrics.json
        - loss.jsonl
        - train.log
    """
    TrainParamsCls = TrainParams
    TrainStatusCls = TrainerStatus
    ControlParamsCls = ControlParams

    ModelCls = nn.Module              # * These four class attribute should be given for the
    HyperparamsCls = Hyperparams      # * trainer, notice the `ModelCls` should be able to 
    DatasetCls = IDataset             # * directly inited from `ModelHyperParamsCls`, in other
    DatasetConfigCls = DatasetConfig  # * words the `__init__` is like `def __init__(self,
    MetricsCls = IMetrics             # * config: ModelHyperParamsCls)`,  for the `DatasetCls`
                                      # * and `Metrics` see the interfaces for details.

    TLogHandlers = ty.Literal['console', 'file']

    # type hinting
    model             : ModelCls
    dataset           : DatasetCls
    train_params      : TrainParamsCls
    hyperparams       : HyperparamsCls
    control_params    : ControlParamsCls

    root_dir          : Path
    status            : TrainStatusCls
    device            : torch.device
    save_dir          : Optional[Path]
    best_valid_metrics: Optional[MetricsCls]   
    current_valid_metrics: Optional[MetricsCls]
    
    def __init__(self, model: ModelCls, hyperparams: HyperparamsCls, train_params: TrainParamsCls, 
                 control_params: ControlParamsCls, root_dir: PathLike, dataset: DatasetCls, 
                 status: Optional[TrainStatusCls] = None) -> None: 

        for name in ['HyperparamsCls', 'DatasetCls', 'MetricsCls', 'ModelCls', 'DatasetConfigCls']:
            if getattr(self, name) is getattr(Trainer, name):
                raise NotImplementedError(f"You should specify the {name} in subclass.")
            
        for func in ['batch_loss', 'get_train_dataloader', 'batch_eval', 'get_eval_dataloader']:
            if getattr(self, func) is getattr(Trainer, func):
                raise NotImplementedError(f"You should implement the method {func} in subclass.")

        if not status:
            status = self.TrainStatusCls(cur_epoch=0, cur_step=0, dataset=str(dataset), 
                                         best_model=None, model_type=self.ModelCls.__name__)

        self.model              = model
        self.dataset            = dataset
        self.train_params       = train_params
        self.hyperparams        = hyperparams
        self.control_params     = control_params
        self.root_dir           = Path(root_dir)
        self.status             = status
        
        self.device             = torch.device('cpu')
        self.save_dir           = None
        self.best_valid_metrics = None
        self.train_losses       = None
        self.loss_file          = None
        self.test_metrics       = None
        self.train_begin_time   = None

        # for validation checking in the training loop
        self.current_valid_metrics = None

        self.optimizer, self.scheduler, self.train_loader = None, None, None
        self.loss_value = None
        self.best_model, self.best_epoch, self.best_step = status.best_model,\
            status.best_epoch, status.best_step

        self.logger_handles: dict[self.TLogHandlers, logging.Handler] = {}
        self.logger = None
        self.create_logger()

        # TODO: Train Event Map: (event, control_params) -> Callable

    @classmethod
    def from_checkpoint(cls, save_dir: PathLike, model = None, dataset = None,
                        epoch=None, step=None, **kargs) -> 'Trainer':
        r""" init trainer from saved checkpoint, notice by default trainer would load the
        best_model record in status.

        Args:
            `save_dir`: the directory path of the checkpoint
            `model`: default init model, if not given will try to init the model using saved 
                hyperparams(`hyperparams.json`)
            `dataset`: default init dataset, if not given will try to init the dataset using 
                saved config(`dataset.json`)
            `kargs`: these key args would pass to the constructor of cls.
        """
        save_dir = Path(save_dir)
        root_dir = save_dir.parent
        if not save_dir.exists():
            raise FileNotFoundError(f"{save_dir} not exists")
        status = json_load_dataclass(save_dir/STATUS_FILE_NAME, cls.TrainStatusCls)
        train_params = json_load_dataclass(save_dir/TRAIN_PARAMS_FILE_NAME, cls.TrainParamsCls)
        if dataset is None:
            dataset = cls.DatasetCls.from_config(json_load_dict(save_dir/DATASET_FILE_NAME))
        if model is None:
            hyperparams = json_load_dataclass(save_dir/HYPERPARAMS_FILE_NAME, cls.HyperparamsCls)
            model = cls.ModelCls(hyperparams)
        control_params = json_load_dataclass(save_dir/CONTROL_PARAMS_FILE_NAME, 
                                             cls.ControlParamsCls)
        trainer = cls(model, hyperparams, train_params, control_params, root_dir, dataset, 
                      status=status, **kargs)
        
        if epoch is not None and step is not None:
            raise ValueError("You should only specify one of epoch and step")
        
        load_model_dir = None
        if epoch:
            load_model_dir = save_dir/'models'/f'epoch_{epoch}'
        elif step:
            load_model_dir = save_dir/'models'/f'step_{step}'
        elif status.best_model:
            load_model_dir = save_dir/'models'/status.best_model
        else:
            raise ValueError("You should specify the epoch or step to load the model "
                             "if the best_model not exists.")
        
        trainer.load_model(load_model_dir)
        if epoch:
            trainer.status.cur_epoch+=1
        elif step:
            trainer.status.cur_step+=1
        trainer.save_dir = save_dir
        if (save_dir/BEST_VALID_METRICS_FILE_NAME).exists():
            trainer.best_valid_metrics = cls.MetricsCls.load(save_dir/BEST_VALID_METRICS_FILE_NAME)
        return trainer

    def train(self, train_log_level = logging.DEBUG) -> MetricsCls:

        # * 1. preparing stored folder, save files
        if self.save_dir is None:  # if the trainer is from checkpoints, save_dir is already set
            self.save_dir = self.get_save_dir()
            self.save_dir.mkdir(exist_ok=True, parents=True)

        (self.save_dir/'models').mkdir(exist_ok=True, parents=True)
        (self.save_dir/'validations').mkdir(exist_ok=True, parents=True)

        resumed = self.check_resume()  # check resume and resume model, status, optimizer ...

        log_file = self.save_dir/TRAIN_LOG_FILE_NAME  # save training log
        formatter = self.default_logger_format()
        file_handle = self.logger_handles.get('file',
                                              logging.FileHandler(log_file, mode="a", encoding="utf-8"))
        file_handle.setFormatter(formatter)
        file_handle.setLevel(train_log_level)
        self.logger.addHandler(file_handle)

        if resumed:
            self.logger.info("resumed")
        else:
            self.train_begin_time = datetime.now()  # use to estimate end time
            # save train_params and model hyperparams
            json_dump_dataclass(self.train_params, self.save_dir/TRAIN_PARAMS_FILE_NAME)
            json_dump_dict(self.dataset.dump_config(), self.save_dir/DATASET_FILE_NAME)
            json_dump_dataclass(self.hyperparams, self.save_dir/HYPERPARAMS_FILE_NAME)
            json_dump_dataclass(self.control_params, self.save_dir/CONTROL_PARAMS_FILE_NAME)
            self.logger.info(f"save to {self.save_dir}")
            self.status.cur_epoch = 1  # TODO: ???, figure out how to optimize this
            self.status.cur_step  = 1

        # * 2. preparing optimizer, scheduler and dataloader, change model status
        self.create_optimizer_lr_scheduler_maybe()  # 
        self.train_loader = self.get_train_dataloader()
        self.model.to(self.device)

        # * 3. the training - valid loop
        self.train_loop()

        # * 4. test the best model
        self.logger.info(f"load best model {self.best_model}.")
        self.load_model(self.save_dir/'models'/self.best_model)
        self.test_metrics = self.test()

        # * 5. save & exit
        self.save_status()
        self.close_loss_logger()
        self.test_metrics.save(self.save_dir/TEST_METRICS_FILE_NAME)
        return self.test_metrics  # return the test result as final train result
    
    def create_optimizer_lr_scheduler_maybe(self, force = False):
        if self.optimizer is None or force:
            optimizer_cls = getattr(torch.optim, self.train_params.optimizer_cls)
            param_groups = self.prepare_param_groups()
            lr, wd = self.train_params.lr, self.train_params.weight_decay
            self.optimizer = optimizer_cls(param_groups, lr=lr, weight_decay=wd)
            self.optimizer.zero_grad()  # actually not needed

            self.scheduler = None
            if self.train_params.lr_scheduler:
                try:
                    self.scheduler = self.prepare_scheduler()
                except AttributeError:
                    self.logger.warning(f"Scheduler {self.train_params.lr_scheduler}  not found, "
                                        "using None scheduler.")

    def check_resume(self):
        r""" check if the trainer is resuming from the last checkpoint. """
        resume_flag = False
        ckpt_dir = None

        if self.control_params.loop_control == 'epoch':
            if self.status.cur_epoch > 1:
                resume_flag = True
                last_saved_epoch = ((self.status.cur_epoch-1)//self.control_params.save_span_epoch)\
                                    * self.control_params.save_span_epoch
                ckpt_dir = self.save_dir/'models'/f'epoch_{last_saved_epoch}'
        elif self.control_params.loop_control == 'step':
            if self.status.cur_step > 1:
                resume_flag = True
                last_saved_step = ((self.status.cur_step-1)//self.control_params.save_span_steps)\
                                    * self.control_params.save_span_step
                ckpt_dir = self.save_dir/'models'/f'step_{last_saved_step}'

        if resume_flag:
            self.create_optimizer_lr_scheduler_maybe(True)
            self.load_model(ckpt_dir)
            elapsed_time = timedelta(seconds=float(self.status.training_seconds))
            self.train_begin_time = datetime.now() - elapsed_time
            
        return resume_flag

    def linear_scheduler_with_warmup(self):
        r""" return a linear wramup schduler from min_lr to lr. """
        num_steps = self.train_params.num_steps
        warmup_proportion = self.train_params.warmup_proportion
        lr, min_lr = self.train_params.lr, self.train_params.min_lr

        warmup_steps = num_steps * warmup_proportion
        warmup_incr = (lr - min_lr) / warmup_steps
        decr = (lr - min_lr) / (num_steps - warmup_steps)
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                gamma = (min_lr + current_step * warmup_incr)/lr
            else:
                gamma = (lr - decr * (current_step - warmup_steps))/lr
            return gamma
        return LambdaLR(self.optimizer, lr_lambda)
    
    def expotional_lr_scheduler(self):
        return torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 
                                                      gamma=self.train_params.lr_decay)
    
    def prepare_scheduler(self):
        return getattr(self, self.train_params.lr_scheduler)()
    
    def valid_epoch(self, epoch = None):
        r""" validation according to epoch. """
        if epoch is None:
            epoch = self.status.cur_epoch
        valid_metrics = self.valid()
        self.logger.debug(f"epoch {epoch} valid metric: {valid_metrics}")
        valid_metrics.save(self.save_dir/'validations'/f'epoch_{epoch}.json')
        
        if valid_metrics > self.best_valid_metrics:
            self.best_valid_metrics = valid_metrics
            self.best_valid_metrics.save(self.save_dir/BEST_VALID_METRICS_FILE_NAME)
            self.best_model = f'epoch_{epoch}'
            self.best_epoch = epoch
            self.save_status()

    def valid_step(self, step = None):
        r""" validation according to step. """
        if step is None:
            step = self.status.cur_step
        valid_metrics = self.valid()
        self.logger.debug(f"step {step} valid metric: {valid_metrics}")
        valid_metrics.save(self.save_dir/'validations'/f'step_{step}.json')
        
        if valid_metrics > self.best_valid_metrics:
            self.best_valid_metrics = valid_metrics
            self.best_valid_metrics.save(self.save_dir/BEST_VALID_METRICS_FILE_NAME)
            self.best_model = f'step_{step}'
            self.best_step = step
            self.save_status()

    def to(self, device):
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def save_status(self) -> None:
        self.status.training_seconds = int((datetime.now() - self.train_begin_time).seconds)
        self.status.best_model = self.best_model
        self.status.best_epoch = self.best_epoch
        self.status.best_step  = self.best_step
        json_dump_dict(asdict(self.status), self.save_dir/STATUS_FILE_NAME)

    def should_early_stop(self) -> bool:
        r""" return True if metric * prob < best_metric """
        if self.control_params.early_stop_prop is not None:
            best_valid = getattr(self, 'best_valid_metrics', None)
            cur_valid = getattr(self, 'current_valid_metrics', None)
            if best_valid:
                best_metric = float(best_valid)
                cur_metric = float(cur_valid)
                if cur_metric < best_metric * self.control_params.early_stop_prop:
                    return True

        return False

    def stop_training_epoch(self):
        return (self.status.cur_epoch > self.train_params.num_epochs or
                self.should_early_stop())
    
    def stop_training_step(self):
        return (self.status.cur_step > self.train_params.num_steps or
                self.should_early_stop())

    def stop_training(self):
        r""" default stop training judge, would be override """
        if self.control_params.loop_control == 'epoch':
            return self.stop_training_epoch()
        elif self.control_params.loop_control == 'step':
            return self.stop_training_step()

    def train_loop(self):
        r""" training loop, notice here we except test """
        self.prepare_train()
        self.prepare_loss_log()
        grad_acc_counter = 0
        self.loss_value = 0.
        self.optimizer.zero_grad()
        while not self.stop_training(): 
            self.prepare_epoch()
            for microbatch in self.train_loader:
                self.prepare_step()
                self.loss_value += self.forbackward_step(microbatch)  # loss compute, backward
                grad_acc_counter += 1
                # grad_accmulate happend if ...
                if grad_acc_counter % self.control_params.grad_acc_step == 0:
                    grad_acc_counter = 0
                    self.optimize_step()
                    self.step_end()  # step++ toggle validation, save, etc
                    self.loss_value = 0.
                    if self.stop_training():
                        break
            self.epoch_end()  # epoch ++

        self.after_train()  # final validation by default

    def prepare_train(self):
        r""" preparation before the actual training loop, default is doing nothing. """

    def prepare_loss_log(self):
        r"""  by default using losses.jsonl for loss log. """
        loss_file_path = self.save_dir/'losses.jsonl'
        loss_file = loss_file_path.open('w')
        loss_file.seek(0, 1)
        self.loss_file = loss_file
        self.train_losses = []

    def estimate_time(self):
        r""" estimate the time of the training """
        if getattr(self, 'train_begin_time', None) is None:
            # * here avoid stopping training.
            return None
        
        if self.control_params.loop_control == 'epoch':
            time_passed = datetime.now() - self.train_begin_time
            time_per_epoch = time_passed / self.status.cur_epoch
            time_left = time_per_epoch * (self.train_params.num_epochs - self.status.cur_epoch)
            return time_left
        elif self.control_params.loop_control == 'step':
            time_passed = datetime.now() - self.train_begin_time
            time_per_step = time_passed / self.status.cur_step
            time_left = time_per_step * (self.train_params.num_steps - self.status.cur_step)
            return time_left

    def log_loss(self):
        r""" log the loss to file, including the loss mean control. """
        self.train_losses.append(self.loss_value)

        log = False
        if self.control_params.loss_log_control == 'step':
            if self.status.cur_step % self.control_params.loss_log_steps == 0:
                log = True
        
        elif self.control_params.loss_log_control == 'epoch':
            if self.status.cur_epoch % self.control_params.loss_log_epochs == 0:
                log = True

        if log:
            mean_loss = np.mean(self.train_losses)
            msg = ('{ "epoch": %d, "step": %d, "loss": %.4f }' 
                % (self.status.cur_epoch, self.status.cur_step, mean_loss))
            self.loss_file.write(msg + '\n')
            left_time = self.estimate_time()
            if left_time:
                left_time_str = "{:02}:{:02}:{:02}".format(left_time.seconds//3600,\
                                                           (left_time.seconds//60)%60,\
                                                           left_time.seconds%60)
                msg += f", time left: {left_time_str}."
            self.logger.info(msg)
            self.train_losses = []
        
    def log_metrics(self, msg, metrics):
        r""" log metrics, default is using `logger.info` """
        self.logger.info(f"{msg}: {metrics}")

    def close_loss_logger(self):
        r""" close the loss logger, by default close the `self.loss_file` """
        self.loss_file.close()

    def prepare_step(self) -> Any:
        r""" 
        prepare things for a single train, should return a minibatch. Return None will stop 
        the training loop (but still run `self.step_end`).
        """
        self.save_status()

    def prepare_param_groups(self):
        r""" default prepare param groups """
        return [{
            'params': self.model.parameters(),
            'initial_lr': self.train_params.lr  # for scheduler resume
        }]

    def after_train(self):
        r""" default after train: close_loss_logger, validate in between checkpoints """
        self.close_loss_logger()
        if self.control_params.valid_after_train:
            self.after_train_valid()
        self.final_validation()

    def after_train_valid(self):
        r""" to support not valid during train, doing the valid after finish training. """
        end = (self.train_params.num_epochs
               if self.control_params.loop_control == 'epoch' 
               else self.train_params.num_steps)
        span = (self.control_params.valid_span_epoch
                if self.control_params.loop_control == 'epoch' 
                else self.control_params.valid_span_steps)
        begin = span
        unit = self.control_params.loop_control
        valid_func = (self.valid_epoch 
                      if self.control_params.loop_control == 'epoch' 
                      else self.valid_step)
        self.logger.info(f"validing models from {begin} to {end} with span {span}")
        for i in range(begin, end+1, span):
            self.load_model(self.save_dir/'models'/f'{unit}_{i}')
            self.logger.info(f"validing model from {unit} {i}")
            valid_func(i)

    def final_validation(self):
        r""" find the best saved checkpoint """
        # if by epoch,
        if self.control_params.loop_control == 'epoch':
            best_epoch, save_epochs = self.best_epoch, self.control_params.save_span_epoch
            valid_span, cur_epoch = self.control_params.valid_span_epoch, self.status.cur_epoch

            begin_epoch = max(save_epochs*2, best_epoch - valid_span + save_epochs)
            end_epoch = min(cur_epoch, best_epoch + valid_span) 
            for epoch in range(begin_epoch, end_epoch, save_epochs):
                if epoch == best_epoch:
                    continue
                self.logger.info(f"final validation for epoch {epoch}")
                self.load_model(self.save_dir/'models'/f'epoch_{epoch}')
                self.valid_epoch(epoch)
        # if by steps,
        elif self.control_params.loop_control == 'step':
            cur_step, best_step = self.status.cur_step, self.best_step
            valid_span_steps, save_steps = self.control_params.valid_span_steps, \
                                           self.control_params.save_span_steps
            
            begin_step = max(save_steps*2,  best_step - valid_span_steps + save_steps)
            end_step = min(cur_step, best_step + valid_span_steps)
            for step in range(begin_step, end_step, save_steps):
                if step == best_step:
                    continue
                self.logger.info(f"final validation for step {step}")
                self.load_model(self.save_dir/'models'/f'step_{step}')
                self.valid_step(step)
        else:
            raise ValueError("Unknown loop control")
        
    class CustomFormatter(logging.Formatter):
        def __init__(self, fmt = None, datefmt = None, style = "%", validate = True, *,
                     defaults = None, trainer: 'Trainer' = None):
            super().__init__(fmt, datefmt, style, validate, defaults=defaults)
            self.trainer = trainer
            self.loop_control = trainer.control_params.loop_control
            self.max_loop_num = (trainer.train_params.num_epochs
                                if self.loop_control == "epoch" 
                                else trainer.train_params.num_steps
                                if self.loop_control == "step"
                                else None)
            assert self.max_loop_num is not None
            self.log_length = len(str(self.max_loop_num+1))

        @property
        def loop_num(self):
            trainer = self.trainer
            loop_num = (trainer.status.cur_epoch
                        if self.loop_control == "epoch" 
                        else trainer.status.cur_step
                        if self.loop_control == "step"
                        else None)
            return loop_num
        
        @property
        def loop_num_str(self):
            loop_num_str = str(self.loop_num)
            return ' ' * (self.log_length - len(loop_num_str)) + loop_num_str

        def format(self, record):
            elapsed_time = (datetime.now() - self.trainer.train_begin_time
                            if self.trainer.train_begin_time is not None
                            else datetime.min - datetime.min)
            time_format = (datetime.min + elapsed_time).strftime("%H:%M:%S")
            record.elapsed_time = time_format
            record.train_loop = f"{self.loop_control} " + self.loop_num_str
            return super().format(record)

    def default_logger_format(self):
        return self.CustomFormatter("%(elapsed_time)s %(train_loop)s: %(message)s",
                                    trainer=self)

    def create_logger(self, level = logging.INFO):
        # ERROR -> CRITICAL -> WARNING -> INFO -> DEBUG
        if self.logger is not None:
            self.logger.setLevel(level)
            return
        logger = logging.getLogger(type(self).__name__)
        logger.setLevel(level)
        console_handler = self.logger_handles.get('console', logging.StreamHandler())
        console_handler.setLevel(level)
        formatter = self.default_logger_format()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        self.logger = logger
        self.logger_handles['console'] = console_handler
    
    def set_logger_level(self, level: int, handler: TLogHandlers = 'console'):
        self.logger_handles[handler].setLevel(level)

    TMicroBatch = ty.Union[ty.Dict[str, torch.Tensor], ty.List[torch.Tensor],
                           ty.Tuple[torch.Tensor], torch.Tensor]

    def move_single_batch(self, batch: TMicroBatch) -> TMicroBatch:
        r""" 
        move input minibatch to self.device, could only be done for `Sequence[Tensor]`,
        `dict[str, Tensor]` and `Tensor`
        """
        if isinstance(batch, torch.Tensor):
            batch = batch.to(self.device)
        elif isinstance(batch, Sequence):
            batch = [(x.to(self.device) if isinstance(x, torch.Tensor) else x)
                          for x in batch]
        elif isinstance(batch, dict):
            batch = { k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) 
                      for k, v in batch.items() }
        else:
            self.logger.warning(f"Unrecognized minibatch type {type(batch).__name__}, "
                                f"can't moved to device {str(self.device)}.")
        return batch

    def forbackward_step(self, microbatch: TMicroBatch):
        microbatch = self.move_single_batch(microbatch)
        loss = self.batch_loss(microbatch)
        assert isinstance(loss, torch.Tensor), "The return of train_batch should be a torch.Tensor"
        if self.control_params.grad_acc_step > 1:
            loss = loss / self.control_params.grad_acc_step
        loss.backward()
        # maybe gradient accumulation happened
        return loss.item()
    
    def optimize_step(self):
        self.before_optimize()
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

    def before_optimize(self):
        r""" hook before `optimizer.step` after `loss.backward`, default is do nothing. """

    def get_train_dataloader(self):
        r"""
        this method is calling for the train dataloader, notice the beginning of a new epoch is
        got by the `Stop`
        """
        raise NotImplementedError("This method get_train_dataloader should be implemented in "
                                  "subclass")
    
    def batch_loss(self, microbatch: TMicroBatch) -> torch.Tensor:
        r""" interface, should return a loss tensor """
        raise NotImplementedError("This method train_batch should be implemented in subclass")
    
    def prepare_epoch(self):
        r""" 
        preparation called before a new epoch, default is doing nothing, begin of an
        epoch by default is when `train_dataloader` throws the `StopIteration` exception. 
        """

    def epoch_end(self):
        if self.control_params.loop_control == 'epoch':
            self.log_loss()
            if self.status.cur_epoch % self.control_params.save_span_epoch == 0:
                self.save_model(self.save_dir/'models'/f'epoch_{self.status.cur_epoch}')
            if (self.status.cur_epoch % self.control_params.valid_span_epoch == 0 and
                (not self.control_params.valid_after_train or self.stop_training())):
                self.valid_epoch()
        self.status.cur_epoch += 1

    def step_end(self):
        if self.control_params.loop_control == 'step':
            self.log_loss()
            if self.status.cur_step % self.control_params.save_span_steps == 0:
                self.save_model(self.save_dir/'models'/f'step_{self.status.cur_step}')
            if (self.status.cur_step % self.control_params.valid_span_steps == 0 and
                (not self.control_params.valid_after_train or self.stop_training())):
                    self.valid_step()
        self.status.cur_step += 1
        self.train_losses.append(self.loss_value)

    def get_eval_dataloader(self, is_valid: bool) -> Any:
        r""" interface, should return a dataloader """
        raise NotImplementedError("This method get_eval_dataloader should be implemented in"
                                  "subclass")
    
    def prepare_eval(self, is_valid: bool, dataloader, metrics):
        r""" preparation before eval, default is do nothing. """

    def after_eval(self, metrics):
        r""" after eval, default is return pass in metrics, notice the metrics.average() 
             has been called. 
        """
        if not self.stop_training():
            self.model.train()
        return metrics

    def init_metrics(self) -> MetricsCls:
        r""" init metrics, default is using the None parameter constructor."""
        return self.MetricsCls()

    @torch.no_grad()
    def eval(self, is_valid: bool) -> MetricsCls:
        r""" interface, should return a MetricsCls """
        dataloader = self.get_eval_dataloader(is_valid)
        self.model.eval()
        metrics = self.init_metrics()
        self.prepare_eval(is_valid, dataloader, metrics)
        for minibatch in dataloader:
            minibatch = self.move_single_batch(minibatch)
            metrics_data = self.batch_eval(minibatch)
            metrics.accumulate(*metrics_data)
        metrics.average()
        metrics = self.after_eval(metrics)
        return metrics

    def batch_eval(self, minibatch) -> ...:
        r""" interface, should return the input params for `MetricsCls.accumulate`. """
        raise NotImplementedError("This method batch_eval should be implemented in subclass")

    def test(self) -> MetricsCls:
        test_metrics = self.eval(False)
        assert test_metrics, "test metrics should not be None"
        self.test_metrics = test_metrics
        self.logger.info(f"test metrics: {str(test_metrics)}")
        return test_metrics

    def valid(self) -> MetricsCls:
        valid_metrics = self.eval(True)
        assert valid_metrics, "valid metrics should not be None"
        self.current_valid_metrics = valid_metrics
        self.logger.info(f"valid metrics: {str(valid_metrics)}")
        return valid_metrics

    def save_model(self, save_path: PathLike) -> None:
        r""" default save model is save the model state_dict and optimizer state_dict """
        if not isinstance(save_path, Path):
            save_path = Path(save_path)
        save_path.mkdir(exist_ok=True, parents=True)
        torch.save(self.model.state_dict(), save_path/MODEL_CKPT_FILE)
        json_dump_dataclass(self.status, save_path/STATUS_FILE_NAME)
        if self.optimizer:
            torch.save(self.optimizer.state_dict(), save_path/OPTIMIZER_CKPT_FILE)

    def load_model(self, load_path: PathLike) -> None:  # nn.Module:
        r""" 
        default load model is using model directly load the save state_dict, also optimizer,
        take care here self.model must be inited 
        """
        if not isinstance(load_path, Path):
            load_path = Path(load_path)
        if self.model is None:
            raise ValueError("You should init the model before loading")
        model_params = torch.load(load_path/'pytorch_model.bin', map_location=self.device,
                                  weights_only=True)
        self.model.load_state_dict(model_params)
        if (load_path/STATUS_FILE_NAME).exists():
            self.status = json_load_dataclass(load_path/STATUS_FILE_NAME, self.TrainStatusCls)
        if self.optimizer:
            optimizer_params = torch.load(load_path/OPTIMIZER_CKPT_FILE, map_location=self.device,
                                          weights_only=True)
            self.optimizer.load_state_dict(optimizer_params)

    def get_save_dir(self) -> Path:
        r""" return the generate save_dir, notice this won't create the directory. """
        dir_name = f'{str(self.dataset)}-{type(self.model).__name__}_'
        dir_name += datetime.now().strftime("%m-%d-%H:%M:%S")
        save_dir = self.root_dir/dir_name
        return save_dir

    # * begin utils
    CKPT_keys = Literal['status', 'dataset', 'test_metrics', 'hyperparams', 'train_params', 
                        'control_params']
    CKPTS = list[dict[CKPT_keys, Any]]
    CKPT_MAP = dict[CKPT_keys, Optional[dict[Any, set[int]]]]
    
    @classmethod
    def sorted_ckpts(cls, idxes: set[int], statics: CKPTS):
        idxes = list(idxes)
        return sorted(idxes, key=lambda x: statics[x]['test_metrics'], reverse=True)

    @classmethod
    def read_validations(cls, checkpoint: PathLike) -> list[dict]:
        r""" read all the validation jsons in the checkpoint """
        checkpoint = Path(checkpoint)
        valid_paths = list(checkpoint.glob('validations/*'))
        valid_metrics = [ json_load_dict(valid_path) for valid_path in valid_paths ]
        unit = valid_paths[0].stem.split('_')[0]
        numbers = [ int(p.stem.split('_')[-1]) for p in valid_paths ]
        valid_metrics = sorted(zip(numbers, valid_metrics), key=lambda x: x[0])
        return unit, valid_metrics

    @classmethod
    def read_losses(cls, checkpoint: PathLike) -> list[dict]:
        checkpoint = Path(checkpoint)
        loss_file = checkpoint/'losses.jsonl'
        with open(loss_file, 'r', encoding="UTF-8") as f:
            lines = f.readlines()
        losses = [ json.loads(line) for line in lines ]
        return losses
