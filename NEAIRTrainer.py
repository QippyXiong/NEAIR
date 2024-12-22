import math
from os import PathLike
from dataclasses import dataclass
import typing as ty
import functools

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.torch_version
import torch.utils
import torch.utils.data

from common.interfaces import TrainParams, TrainerStatus, ControlParams
from common.trainer import Trainer
from common import utils

from neair import NEAIR, NEAIRConfig
from loader import InducKGDataset, InducKGDatasetConfig
from metrics import KGCMetrics


@dataclass(unsafe_hash=True)
class NEAIRTrainParams(TrainParams):
    lr: float                = 5e-3
    min_lr: float            = 5e-4
    batch_size: int          = 100 
    weight_decay: float      = 0.  
    dropout: float           = 0.0 
                              
    optimizer_cls: str       = 'AdamW'
    lr_scheduler: str        = ''
    warmup_proportion: float = 0.0
    lr_decay: float          = 1.0
    dataset: str             = ''
    loss_func: str           = 'multiclslog_loss'
    num_epochs: int          = 50
    num_steps: int           = -1
    num_negs: int            = 0xfffffff
    resplit_span_epoch: int  = 1


@dataclass(unsafe_hash=True)
class NEAIRControlParams(ControlParams):
    loop_control: str     = 'epoch'
    valid_span_epoch: int = 1
    save_span_epoch : int = 1

    valid_span_steps: int = -1  # 4000
    save_span_steps : int = -1  # 1000

    eval_batch_size: int  = 32

    loss_log_steps:   int = 100
    loss_log_epochs:  int = 1


class NEAIRTrainer(Trainer):
    TrainParamsCls = NEAIRTrainParams
    TrainStatusCls = TrainerStatus
    ControlParamsCls = NEAIRControlParams
    
    ModelCls = NEAIR                 
    HyperparamsCls = NEAIRConfig
    DatasetCls = InducKGDataset
    DatasetConfigCls = InducKGDatasetConfig
    MetricsCls = KGCMetrics

    # type hinting
    model: NEAIR
    dataset: InducKGDataset
    train_params: NEAIRTrainParams

    def __init__(self, model: NEAIR, hyperparams: NEAIRConfig, train_params: TrainParams, 
                 control_params: ControlParams, root_dir: PathLike, 
                 dataset: InducKGDataset,
                 status: ty.Optional[TrainerStatus] = None, 
                 Ks: ty.Optional[list[int]]=None, test_while_valid = False,
                 use_optimized_model = False) -> None:
        super().__init__(model, hyperparams, train_params, control_params, root_dir, dataset,
                         status)
        Ks = Ks or [1,3,10]

        self.loss_func = getattr(self, self.train_params.loss_func)
        self.Ks = Ks
        if not model.loader:
            model.loader = dataset
        self.test_while_valid = test_while_valid

        if hasattr(model, 'set_dropout'):
            self.model.set_dropout(self.train_params.dropout)

        if dataset.config.induc:
            self.ind_dataset = dataset.ind_dataset

        self.dataset = dataset.dataset

        if torch.__version__ >= (2,5,0) and use_optimized_model:
            self.model = torch.compile(self.model)
            torch.set_float32_matmul_precision('high')
        elif use_optimized_model:
            self.logger.warning("Cannot use optimized model because your pytorch version is"
                                f"{torch.__version__}, less than "
                                f"{torch.torch_version.TorchVersion('2.5.0')}")

    @classmethod
    @functools.wraps(Trainer.from_checkpoint)
    def from_checkpoint(cls, *args, **kargs) -> 'NEAIRTrainer':
        trainer = super().from_checkpoint(*args, **kargs)
        trainer.model.loader = trainer.dataset
        return trainer

    # * --- train ---
    # @override
    def get_train_dataloader(self):
        batch_size = self.train_params.batch_size // self.control_params.grad_acc_step
        loader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, 
                                             shuffle=False, drop_last=True, num_workers=1)
        # NOTE: every resplit would shuffle the train dataset so shuffle here is unnecessary.
        return iter(loader)

    # @override
    def prepare_train(self):
        # need to move to train dataset
        super().prepare_train()
        self.model.loader = self.dataset

    # @override
    def prepare_epoch(self):
        super().prepare_epoch()
        restore_before_merged_facts =\
            (self.status.cur_epoch % self.train_params.resplit_span_epoch == 0
             and self.train_params.resplit_span_epoch > 1)
        self.dataset.split_facts_train(restore_before_merged_facts)
        self.train_loader = self.get_train_dataloader()

    # @override
    def batch_loss(self, microbatch) -> Tensor:
        triplets, labels = microbatch
        heads, rels, tails = triplets[:,0], triplets[:,1], triplets[:,2]

        scores = self.model.forward(heads, rels)
        loss = self.loss_func(scores, labels, targets=tails)
        return loss
    
    def should_early_stop(self) -> bool:
        # * NOTE: if after 2 epochs, the valid metrics is too low, stop
        # * if set valid_span_epoch > 2, float would occur bugs
        cur_valid_metrics = self.current_valid_metrics or 1.0
        should_stop_too_low = (self.status.cur_epoch > 2 and
                               float(cur_valid_metrics) < 0.1)

        return super().should_early_stop() or should_stop_too_low
        
    # * --- eval ---
    # @override
    def init_metrics(self):
        return KGCMetrics(Ks=self.Ks, compute_coverage=True)

    # @override
    def get_eval_dataloader(self, is_valid):        
        ds = (self.ind_dataset.get_evaluate_dataset(is_valid) 
              if getattr(self, 'ind_dataset', None) is not None and not is_valid else
              self.dataset.get_evaluate_dataset(is_valid))

        batch_size=self.control_params.eval_batch_size
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    # @override
    def prepare_eval(self,  is_valid: bool, dataloader, metrics):
        # super().prepare_eval()
        if getattr(self, 'ind_dataset', None) is not None:
            if is_valid:  # validation using the training knowledge graph
                self.dataset.merge_facts()
                self.model.loader = self.dataset
            if not is_valid:  # for testing we select the inductive testing graph
                self.model.loader = self.ind_dataset
        else:
            self.dataset.merge_facts()

    def eval(self, is_valid: bool):
        # dataloader = self.get_eval_dataloader(is_valid)
        if self.test_while_valid and is_valid:
            t = super().eval(False)
            self.logger.info(f"test result: {t}")
        return super().eval(is_valid)

    @functools.wraps(Trainer.after_eval)
    def after_eval(self, *args, **kargs):
        result = super().after_eval(*args, **kargs)
        self.model.loader = self.dataset
        return result

    def test(self):
        with utils.TimeRecoder('test', self.logger.info):
            return super().test()

    # @override
    def batch_eval(self, minibatch):
        # N_e = self.dataset.N_e if getattr(self, 'ind_dataset', None) is None\
        #       else self.ind_dataset.N_e
        triplets, filter_bias = minibatch
        h_ids, r_ids, t_ids = triplets[:, 0], triplets[:, 1], triplets[:, 2]

        scores = self.model.forward(h_ids, r_ids)

        return scores, t_ids, filter_bias, True

    def get_num_negs(self):
        return self.train_params.num_negs

    # * begin loss functions
    def bce_loss(self, pred_scores: Tensor, label: Tensor, targets: Tensor,
                 *args, **kwargs) -> Tensor:
        batch_r = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        neg_ent_ids = self.strict_neg_sample(label, self.train_params.num_negs)
        
        loss_ent_ids = torch.cat((targets[:, None], neg_ent_ids), dim=1)
        loss_label = label[batch_r[:, None], loss_ent_ids]
        loss_scores = pred_scores[batch_r[:, None], loss_ent_ids]
        loss = F.binary_cross_entropy_with_logits(loss_scores, loss_label)
        return loss
    
    @staticmethod
    def strict_neg_sample(label: Tensor, num_negs: int):
        sort_weights = torch.zeros(label.shape, dtype=torch.float32, 
                                       device=label.device).uniform_()
        # * the lower precision for float the more unaverage would appear, but float32 
        # * performs almost same with float64 and with `torch.randint`
        sort_weights.masked_fill_(label, -1e3)
        _, neg_ent_ids = torch.topk(sort_weights, num_negs, dim=1, sorted=False)
        return neg_ent_ids

    def logsigmoid_loss(self, pred_scores: Tensor, label: Tensor, targets: Tensor, 
                       alpha = 0.0) -> Tensor:
        batch_r = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        pos_scores = pred_scores[batch_r, targets]
        
        num_negs = self.get_num_negs()
        filled_scores = pred_scores.masked_fill(label, -1e7).detach()
        neg_ent_ids = torch.topk(filled_scores, num_negs)[1]
        neg_scores = pred_scores[batch_r[:, None], neg_ent_ids]

        pos_loss = -F.logsigmoid(pos_scores).mean()

        if alpha:
            neg_weight = F.softmax(neg_scores * alpha, dim=-1).detach()
            neg_loss = torch.linalg.vecdot(neg_weight, -F.logsigmoid(-neg_scores))
            neg_loss = neg_loss.mean()
        else:
            neg_loss = -F.logsigmoid(-neg_scores).mean()

        return (pos_loss + neg_loss) / 2

    def logsigmoid_loss_with_mask(self, pred_scores: Tensor, label: Tensor, targets: Tensor, 
                                 alpha = 0.5) -> Tensor:
        r""" `alpha`: adversarial_temperature """
        # neg_scores = pred_scores.masked_fill(label, -1e9)
        batch_r = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        pos_scores = pred_scores[batch_r, targets]
        
        num_negs = self.train_params.num_negs
        neg_ent_ids = torch.randint(0, self.dataset.N_e, (pred_scores.shape[0], num_negs),
                                    device=pred_scores.device)
        neg_scores = pred_scores.masked_fill(label, -1e9)[batch_r[:, None], neg_ent_ids]

        pos_loss = -F.logsigmoid(pos_scores).mean()

        if alpha:
            neg_weight = F.softmax(neg_scores * alpha, dim=-1).detach()
            neg_loss = torch.linalg.vecdot(neg_weight, -F.logsigmoid(-neg_scores))
            neg_loss = neg_loss.mean()
        else:
            neg_loss = -F.logsigmoid(-neg_scores).mean()
        
        return (pos_loss + neg_loss) / 2
    
    def multiclslog_loss(self, pred_scores: Tensor, label: Tensor, targets: Tensor, 
                         *args, **kwargs) -> Tensor:
                         
        b_range = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        pos_scores = pred_scores[b_range, targets]

        num_negs = self.get_num_negs()
        if num_negs < self.dataset.N_e:
            filled_scores = pred_scores.detach().masked_fill(label, -1e7)
            filled_scores[b_range, targets] = 1e7
            topk_idx = torch.topk(filled_scores, num_negs+1)[1]
            pred_scores = pred_scores[b_range[:, None], topk_idx]

        max_n = torch.max(pred_scores, dim=1, keepdim=True)[0]
        loss = torch.mean(-pos_scores + max_n + torch.logsumexp(pred_scores - max_n, dim=1))
        return loss
    
    @staticmethod
    def multiclslog_loss_with_mask(pred_scores: Tensor, label: Tensor, targets: Tensor, 
                                   *args, **kwargs) -> Tensor:
        r""" from RED-GNN """
        b_range = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        max_n = torch.max(pred_scores, dim=1, keepdim=True)[0]

        pos_scores = pred_scores[b_range, targets]
        # here original paper use the whole reaonsing set but we adopt the neg samples
        pos_loss = max_n - pos_scores
        label[b_range, targets] = False
        pred_scores = pred_scores.masked_fill(label, -1e7)  # mask other right answers
        loss = (pos_loss + torch.logsumexp(pred_scores-max_n, dim=1)).mean()
        
        return loss
    
    @staticmethod
    def marginrank_loss(pred_scores: Tensor, label: Tensor, targets: Tensor, *args, 
                        **kwargs) -> Tensor:
        r""" from TransE """
        lf = torch.nn.MarginRankingLoss()
        b_range = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        pos_scores = pred_scores[b_range, targets]
        neg_scores = pred_scores
        target = torch.ones_like(pos_scores)
        return lf.forward(pos_scores, neg_scores, target)

    def infoNCE_loss(self, pred_scores: Tensor, label: Tensor, targets: Tensor,
                     *args, **kwargs) -> Tensor:
        # margin = 5e-2
        # delta = 5e-2
        # margin_tensor = torch.zeros_like(pred_scores, dtype=pred_scores.dtype, 
        #                                  device=pred_scores.device)
        # # margin_tensor[:, 0] = margin
        # pred_scores = pred_scores - margin_tensor
        # pred_scores = pred_scores * alpha
        # # print(pred_scores.shape)
        # margin_tensor[:, 0] = 1.0
        batch_r = torch.arange(pred_scores.shape[0], device=pred_scores.device)
        # neg_ent_ids = self.strict_neg_sample(label, self.train_params.num_negs)
        num_negs = self.get_num_negs()

        filled_scores = pred_scores.masked_fill(label, -1e7).detach()
        filled_scores[batch_r, targets] = 1e7
        topk_idx = torch.topk(filled_scores, num_negs)[1]
        pred_scores = pred_scores[batch_r[:, None], topk_idx]
        loss_label = label[batch_r[:, None], topk_idx]
        
        # loss = F.binary_cross_entropy_with_logits(loss_scores, loss_label.float)
        loss = F.cross_entropy(pred_scores, loss_label.float())
        return loss
