from os import PathLike
from typing import Optional, Sequence

import torch
import numpy as np

from common.interfaces import IMetrics
from common.utils import json_load_dict, json_dump_dict


class KGCMetrics(IMetrics):
    r"""
    Compute MRR, hit@K
    """
    def __init__(self, mrr: Optional[float] = None, hit_rates: Optional[dict[int, float]] = None,
                 Ks: Optional[list[int]] = None, tail_ranks: Optional[Sequence[int]] = None, 
                 head_ranks: Optional[Sequence[int]] = None, compute_coverage: bool = False,
                 missed_score_value: float = 1e-8) -> None:
        super().__init__()
        self.mrr = mrr or 0.
        self.hit_rates = hit_rates if hit_rates is not None else { k: 0. for k in Ks }
        self.tail_ranks = tail_ranks if tail_ranks is not None else np.array([], dtype=np.int32)
        self.head_ranks = head_ranks if head_ranks is not None else np.array([], dtype=np.int32)
        self.Ks = Ks or [1,3,10]
        self.compute_coverage = compute_coverage
        self.missed_score_value = missed_score_value
        if compute_coverage:
            self.missed_total = 0
            self.coverage = 0.

    @property
    def count(self):
        len_tail = len(self.tail_ranks)
        len_head = len(self.head_ranks)
        return max([len_tail, len_head])

    def __mul__(self, scalar: float) -> 'KGCMetrics':
        return KGCMetrics(self.mrr * scalar, {k: v * scalar for k, v in self.hit_rates.items()})

    def __add__(self, metric: 'KGCMetrics') -> 'KGCMetrics':
        return KGCMetrics(self.mrr + metric.mrr, 
                          {k: self.hit_rates[k] + metric.hit_rates[k] for k in self.hit_rates})
    
    def __iadd__(self, metric: 'KGCMetrics') -> 'KGCMetrics':
        self.mrr += metric.mrr
        for k in self.hit_rates:
            self.hit_rates[k] += metric.hit_rates[k]
        return self

    def __truediv__(self, scalar: float) -> 'KGCMetrics':
        return KGCMetrics(self.mrr / scalar, {k: v / scalar for k, v in self.hit_rates.items()})

    def __itruediv__(self, scalar: float) -> 'KGCMetrics':
        self.mrr /= scalar
        for k in self.hit_rates:
            self.hit_rates[k] /= scalar
        return self
    
    def __gt__(self, metric: 'KGCMetrics') -> bool:
        if metric is None:
            return True
        return self.mrr > metric.mrr

    def __lt__(self, metric: 'KGCMetrics') -> bool:
        return not (self > metric)

    def __str__(self) -> str:
        base = f"MRR: {self.mrr:.3f}"
        for k in self.hit_rates:
            base += f", hit@{k}: {self.hit_rates[k]:.3f}"
        if self.compute_coverage:
            base += f", coverage: {self.coverage:.3f}"
        return base
    
    def __float__(self):
        return self.mrr

    @torch.no_grad()
    def accumulate(self, pred_scores: torch.Tensor, targets: torch.Tensor, 
                   filt_bias: torch.Tensor, head_query = True) -> None:
        b_range = torch.arange(len(pred_scores), device=pred_scores.device)

        if self.compute_coverage:
            target_scores = pred_scores[b_range, targets]
            missed = target_scores == 0.
            self.missed_total += missed.sum().item()

        pred_scores.masked_fill_(filt_bias, self.missed_score_value)
        ranks = torch.argsort(torch.argsort(pred_scores, dim=1, descending=True), dim=1) + 1
        ranks = ranks[b_range, targets].float()
        ranks = ranks.cpu().numpy()

        if head_query:
            self.tail_ranks = np.append(self.tail_ranks, ranks)
        else:
            self.head_ranks = np.append(self.head_ranks, ranks)

    def average(self):
        if len(self.head_ranks) <= 1 and len(self.tail_ranks) > 1:
            ranks = self.tail_ranks
        elif len(self.head_ranks) > 1 and len(self.tail_ranks) <= 1:
            ranks = self.tail_ranks
        elif len(self.head_ranks) > 1 and len(self.tail_ranks) > 1:
            ranks = np.concatenate([self.head_ranks, self.tail_ranks])
        else:
            raise ValueError("No data to average.")

        self.mrr = (1. / ranks).mean()
        for K in self.Ks:
            self.hit_rates[K] = (ranks <= K).mean()
        
        if self.compute_coverage:
            self.coverage = (self.count - self.missed_total) / self.count

    def asdict(self):
        return {
            'mrr': self.mrr,
            'hit_rates': self.hit_rates,
            'tail_ranks': self.tail_ranks.tolist(),
            'head_ranks': self.head_ranks.tolist(),
        }
    
    def metric_dict(self):
        mrr = {
            'mrr': self.mrr,
            'coverage': self.coverage
        }
        metrics = mrr
        metrics.update(self.hit_rates)
        return metrics
        
    @classmethod
    def load(cls, save_path: PathLike):
        dict_data = json_load_dict(save_path)
        tail_ranks = np.array(dict_data.get('tail_ranks', []), np.int32)
        head_ranks = np.array(dict_data.get('head_ranks', []), np.int32)
        Ks = list(dict_data['hit_rates'].keys())
        return cls(mrr=dict_data['mrr'], hit_rates=dict_data['hit_rates'], Ks=Ks, 
                   tail_ranks=tail_ranks, head_ranks=head_ranks)

    def save(self, save_path: PathLike) -> None:
        r = self.asdict()
        json_dump_dict(r, save_path)

    @classmethod
    def combine_head_tail(cls, pred_tail: 'KGCMetrics', pred_head: 'KGCMetrics') -> 'KGCMetrics':
        pred_tail.head_ranks = pred_head.head_ranks
        pred_tail.mrr = (pred_tail.mrr + pred_head.mrr) / 2
        for K in pred_tail.hit_rates:
            pred_tail.hit_rates[K] = (pred_tail.hit_rates[K] + pred_head.hit_rates[K]) / 2
        return pred_tail
