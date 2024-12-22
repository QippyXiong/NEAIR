import math
import typing as ty
from dataclasses import dataclass

import torch
from torch import nn, Tensor

from common.interfaces import Hyperparams


class ScatterMultiHeadAttentionOutput(ty.NamedTuple):
    pooled_value: ty.Optional[Tensor] = None
    scaled_value: ty.Optional[Tensor] = None
    attention_weights: ty.Optional[Tensor] = None


def simple_pna(o_shape, e_v, v_scale, v_agg, scalars, aggs, dim=0):
    agg_result = []

    for scalar_type in scalars:
        if scalar_type == 'scaled':
            v = e_v * v_scale.unsqueeze(-1)
        elif scalar_type == 'origin':
            v = e_v
        else:
            raise ValueError(f"Unknown scalar type: {scalar_type}")

        cat_results = []
        sum_agg = None  # avoid duplicate computation for sum and std
        for agg_type in aggs:
            if agg_type == 'min':
                aggv = torch.zeros(o_shape, device=v.device)\
                    .index_reduce_(dim, v_agg, v, 'amin', include_self=False)
            elif agg_type == 'max':
                aggv = torch.zeros(o_shape, device=v.device)\
                    .index_reduce_(dim, v_agg, v, 'amax', include_self=False)
            elif agg_type == 'sum':
                if sum_agg is None:  
                    sum_agg = torch.zeros(o_shape, device=v.device).index_add_(dim, v_agg, v)
                aggv = sum_agg
            elif agg_type == 'mean':
                aggv = torch.zeros(o_shape, device=v.device)\
                    .index_reduce_(dim, v_agg, v, 'mean', include_self=False)
            elif agg_type == 'std':
                sum_agg = torch.zeros(o_shape, device=v.device).index_add_(dim, v_agg, v)
                sq_agg = torch.zeros(o_shape, device=v.device).index_add_(dim, v_agg, 
                                                                          torch.square(v))
                aggv = (sq_agg - sum_agg**2).clamp_min(1e-8).sqrt()
            else:
                raise ValueError(f"Unknown agg type: {agg_type}")

            cat_results.append(aggv)

        agg_result.append(torch.concat(cat_results, dim=-1) if len(cat_results) > 1
                          else cat_results[0])

    agg = (torch.stack(agg_result, dim=-1).flatten(-2, -1)
            if len(scalars) > 1 else agg_result[0])
    return agg


TAttnCompute = ty.Literal['softmax', 'sigmoid']


def scatter_multi_head_attention(q: Tensor, k: Tensor, v: ty.Optional[Tensor], 
                                 query_expand: ty.Optional[Tensor],
                                 value_aggregate: Tensor, 
                                 o_shape: ty.Optional[ty.Sequence[int]] = None,
                                 dim: int = 0, n_heads: int = 1,
                                 compute: TAttnCompute = 'sigmoid',
                                 attn_mask: ty.Optional[Tensor] = None,
                                 *,
                                 return_pooled_value = True,
                                 return_attention_weights = False,
                                 return_scaled_value = False,
                                 pna_aggs = None, 
                                 unmask_aggregate = None,
                                 unmask_shape = None,
                                 softmax_scale = None,
                                 ) -> ScatterMultiHeadAttentionOutput:
    r"""
    Shapes:
        `q`: (N_q, d_q), if 
        `key`: (N_k, d_k)
        `value`: (N_k, d_v)
        `q_idxes`: (N_k), array of related query_idx for every key and value.
        `value_aggregate`: (N_k), idx for final aggregated value.
        `o_shape`: output shape,
        'compute': 'softmax' or 'sigmoid' for attention score
    """    
    assert q.shape[-1] == k.shape[-1], f"q.shape[-1]({q.shape[-1]}) != k.shape[-1]({k.shape[-1]})"

    head_size = k.shape[-1] // n_heads

    if query_expand is not None:
        q = q.index_select(dim, query_expand)

    q = q.unflatten(-1, (n_heads, head_size))
    k = k.unflatten(-1, (n_heads, head_size))
    alphas = torch.linalg.vecdot(q, k, dim=-1) / math.sqrt(head_size)

    if compute == 'softmax':
        if attn_mask is not None:
            alphas = alphas.masked_fill(attn_mask, -1e7)
        weights = torch.exp(alphas)
        idx = unmask_aggregate if unmask_aggregate is not None else value_aggregate
        weight_add = torch.zeros((*unmask_shape, n_heads), device=weights.device)\
            .index_add_(dim, idx, weights)
        if softmax_scale is not None:
            weight_add = weight_add / softmax_scale.unsqueeze(-1)
        weight_add = weight_add.index_select(dim, idx)
        weights = weights / weight_add  # mean

    elif compute == 'sigmoid':
        if attn_mask is not None:

            alphas = alphas.masked_fill(attn_mask, -1e7)
            weights = torch.sigmoid(alphas)
            masked_scale = torch.zeros((*unmask_shape, n_heads), device=weights.device)\
                .index_reduce_(dim, unmask_aggregate, (~attn_mask).float(), 'mean',
                               include_self=False)
            assert not (masked_scale == 0.).any()
            masked_scale = masked_scale.reciprocal_().index_select(dim, unmask_aggregate)
            weights = weights * masked_scale  # expand not maksed values

        else:
            weights = torch.sigmoid(alphas)

    else:
        raise ValueError(f"Unknown compute type: {compute}")

    attn_weights = None
    if return_attention_weights:
        attn_weights = weights

    agg_value = None
    scaled_value = None
    if return_pooled_value or return_scaled_value:
        # scatter reduce to shape (N_q, d)
        v_size = v.shape[-1]
        v = v.unflatten(-1, (n_heads, v_size // n_heads))
        scaled_v = (weights.unsqueeze(-1) * v).flatten(-2, -1)
        if return_scaled_value:
            scaled_value = scaled_v
        if return_pooled_value:
            if compute in {'softmax', 'sigmoid'}:
                agg_value = torch.zeros((*o_shape, v_size), device=q.device)\
                                .index_add_(dim, value_aggregate, scaled_v)
            else:
                raise ValueError(f"Unknown compute type: {compute}")

    return ScatterMultiHeadAttentionOutput(pooled_value=agg_value,
                                           attention_weights=attn_weights,
                                           scaled_value=scaled_value)


class ScatterMultiHeadAttentionLayer(nn.Module):

    def __init__(self, q_size: int, k_size: int, v_size: int, h_size: int, n_heads: int, *,
                 compute: TAttnCompute = 'sigmoid', rel_dropout: float = 0.,
                 proj_q = True, proj_k = False, pna_aggs = None):
        super().__init__()
        self.q_size = q_size
        self.k_size = k_size
        self.v_size = v_size
        self.h_size = h_size
        self.rel_dropout = rel_dropout

        self.n_heads = n_heads
        self.compute = compute
        self.proj_k = proj_k
        self.proj_q = proj_q
        assert k_size % n_heads == 0, f"k_size({k_size}) must be divisible by n_heads({n_heads})"
        assert v_size % n_heads == 0, f"v_size({k_size}) must be divisible by n_heads({n_heads})"

        if proj_q and not proj_k:
            self.attn = nn.Sequential(
                nn.Linear(q_size, k_size, bias=False),
                nn.GELU()
            )
        elif proj_q and proj_k:
            self.q_proj = nn.Linear(self.q_size, self.h_size, bias=False)
            self.k_proj = nn.Linear(self.k_size, self.h_size, bias=False)
        
        self.pna_aggs = pna_aggs

    def forward(self, query: Tensor, key: Tensor, value: Tensor, query_expand: Tensor,
                value_aggregate: Tensor, o_shape: ty.Union[int, ty.Sequence[int]],
                dim: int = 0, *,  
                return_pooled_value = True,
                return_attention_weights = False,
                return_scaled_value = False,
                unmask_aggregate = None,
                unmask_shape = None,
                in_degree = None, in_pow = None) -> ScatterMultiHeadAttentionOutput:
        r"""
        Shapes:
            `q`: (N_q, d_q), if 
            `key`: (N_k, d_k)
            `value`: (N_k, d_v)
            `q_idxes`: (N_k), array of related query_idx for every key and value.
            `value_aggregate`: (N_k), idx for final aggregated value.
            `N_o`: number of output reprsentations, should <= value_aggregate.max() + 1
        """
        if isinstance(o_shape, int):
            o_shape = (o_shape,)

        q, k = None, None
        if self.proj_q and not self.proj_k:
            q = self.attn(query)
            # q = q / math.sqrt(self.q_size)
            k = key
        elif self.proj_q and self.proj_k:
            q = self.q_proj(query)
            # q = q / math.sqrt(self.q_size)
            k = self.k_proj(key)
            # k = k / math.sqrt(self.k_size)
        
        drop_rel_mask = None
        if self.training and self.rel_dropout > 0:
            rel_mask_rand = torch.rand([key.shape[0], self.n_heads], device=key.device)
            drop_rel_mask = rel_mask_rand < self.rel_dropout

            idx = unmask_aggregate
            r_shape = unmask_shape
            max_random_value = torch.zeros((*r_shape, self.n_heads), 
                                            device=rel_mask_rand.device,
                                            dtype=rel_mask_rand.dtype)
            max_random_value.index_reduce_(dim, idx, rel_mask_rand, 'amax',
                                            include_self=False)
            # the max random value relation would stay False
            at_least_one_enable = max_random_value.index_select(dim, idx) >\
                                    rel_mask_rand
            drop_rel_mask = drop_rel_mask & at_least_one_enable

        softmax_scale = None
        if self.compute == 'softmax' and in_degree is not None:
            softmax_scale = ((in_degree+1).log2() if in_pow == -1.0
                             else in_degree.pow(in_pow))

        return scatter_multi_head_attention(q, k, value, query_expand, value_aggregate,
                                            o_shape, dim, self.n_heads, self.compute,
                                            attn_mask=drop_rel_mask,
                                            return_pooled_value=return_pooled_value,
                                            return_attention_weights=return_attention_weights, 
                                            return_scaled_value=return_scaled_value,
                                            pna_aggs=self.pna_aggs,
                                            unmask_aggregate=unmask_aggregate,
                                            unmask_shape=unmask_shape,
                                            softmax_scale=softmax_scale)


class KGEOp(nn.Module):

    def __init__(self, op_type: str):
        super().__init__()
        self.op = getattr(self, op_type)
    
    @staticmethod
    def TransE(ents, rels):
        return ents + rels

    @staticmethod
    def DistMult(ents, rels):
        return ents * rels

    @staticmethod
    def RotatE(ents, rels):
        node_re, node_im = ents.chunk(2, dim=-1)
        edge_re, edge_im = rels.chunk(2, dim=-1)
        message_re = node_re * edge_re - node_im * edge_im
        message_im = node_re * edge_im + node_im * edge_re
        return torch.cat([message_re, message_im], dim=-1)

    def forward(self, ents, rels):
        return self.op(ents, rels)
    

class PNA(nn.Module):

    def __init__(self, hidden_size: int, 
                 scalars: ty.Optional[ty.Set[ty.Literal['scaled', 'origin']]] = None,
                 aggs: ty.Optional[ty.Set[ty.Literal['mean', 'max', 'min', 'std', 'sum']]] = None):
        super().__init__()
        self.d = hidden_size
        if scalars is None:
            scalars = {'scaled'}
        self.scalars = scalars
        self.aggs = aggs or ['mean', 'max', 'min', 'std']
        agg_size = len(self.scalars) * len(self.aggs) * self.d
        self.dropout = nn.Dropout(0.)
        self.proj = nn.Linear(agg_size, hidden_size)

    def set_dropout(self, dropout: float):
        self.dropout.p = dropout

    def forward(self, o_shape, e_v, h_o_degree, h_batch_idx, N_b, h2pair, t2pair) -> torch.Tensor:
        if isinstance(o_shape, int):
            o_shape = (o_shape, self.d)

        assert not (h_o_degree == 0).any(), "ent has outdegree 0"
        h_scale = torch.reciprocal((h_o_degree + 1).log())

        # mean over the whole subgraph
        mean_h_scale_whole = torch.zeros((N_b,), device=e_v.device)\
                                .index_reduce_(0, h_batch_idx, h_scale, reduce='mean',
                                               include_self=False)
        h_scale = h_scale / mean_h_scale_whole.index_select(0, h_batch_idx)

        h_scale = h_scale.index_select(0, h2pair)
        
        agg_value = simple_pna(o_shape, e_v, h_scale, t2pair, self.scalars, self.aggs)
        return self.proj(self.dropout(agg_value))


class GatedResidueNorm(nn.Module):

    def __init__(self, d: int, use_gate = False):
        super().__init__()
        self.d = d
        self.use_gate = use_gate
        self.norm = nn.LayerNorm(d)
        if use_gate:
            self.update_gate = nn.Sequential(
                nn.Linear(3*d, d, bias=False),
                nn.Sigmoid()
            )

    def forward(self, G_h, G_t, q, h_map_t):
        if self.use_gate:
            
            if isinstance(h_map_t, Tensor):
                G_t_has_h = G_t.index_select(0, h_map_t)
                g = self.update_gate(torch.concat([G_t_has_h, G_h, q], dim=-1))
                G_h_updated = G_t_has_h * g + G_h
                G_t.index_copy_(0, h_map_t, G_h_updated)

            elif isinstance(h_map_t, int):
                G_t_has_h = G_t[:h_map_t]
                g = self.update_gate(torch.concat([G_t_has_h, G_h, q], dim=-1))
                G_h_updated = G_t_has_h * g + G_h
                G_t = torch.concat([G_h_updated, G_t[h_map_t:]])

            return self.norm(G_t)

        else:
            if isinstance(h_map_t, Tensor):
                res_idx = h_map_t.unsqueeze(-1).expand(-1, self.d)
                return self.norm(G_t.scatter_add(0, res_idx, G_h))
            
            elif isinstance(h_map_t, int):
                G_h_updated = G_t[:h_map_t] + G_h
                G_t = torch.concat([G_h_updated, G_t[h_map_t:]])
                return self.norm(G_t)


@dataclass
class BfsRepresents:
    E: Tensor
    E_r: Tensor
    r: Tensor               # learning relations
    old_heads_len: int
    ent_pos_map: Tensor     # ent_id => pos in h | pos in G_h
    h2pair: ty.Optional[Tensor]
    t2pair: ty.Optional[Tensor]
    h2rel: ty.Optional[Tensor]
    t2rel: ty.Optional[Tensor]
    pair2rel: Tensor

    e_o_degree: ty.Optional[Tensor] = None
    e_i_degree: ty.Optional[Tensor] = None

    @property
    def h(self): 
        return self.E[:self.old_heads_len]

    @property
    def t(self): 
        return self.E
    
    @property
    def h_o_degree(self): 
        return self.e_o_degree[:self.old_heads_len]
    
    @property
    def t_i_degree(self):
        return self.e_i_degree


@dataclass
class NEAIRConfig(Hyperparams):
    embed_size: int  = -1
    max_depth: int   = -1
    n_heads_rel: int = -1
    rel_attn: TAttnCompute = 'sigmoid'
    ent_attn: ty.Optional[TAttnCompute] = None
    n_heads_ent: int = 0
    n_rels: int = -1
    attn_size: int = 0

    trans_op: ty.Literal['TransE', 'DistMult', 'RotatE'] = 'DistMult'

    use_gate: bool = True
    variant: ty.Literal['SingleAttn', 'SinglePNA', 'DoubleAttn', 'AttnPNA'] = 'AttnPNA'
    rel_dropout: float = 0.
    use_original_scalar_pna: bool = False
    use_sum_replace_pna: bool = False
    in_degree_highlight: float = -1.0 # 0.

    @property
    def n_heads(self):
        return self.n_heads_rel

    @n_heads.setter
    def n_heads(self, n: int):
        self.n_heads_rel = n
        self.n_heads_ent = n
    
    @property
    def r_size(self):
        if self.rel_attn == 'sigmoid_pna':
            return 4 * self.embed_size
        if self.rel_attn == 'sigmoid_minmax':
            return self.embed_size * 2
        return self.embed_size
    
    @property
    def pna_aggs(self):
        if self.use_sum_replace_pna:
            return ['sum']
        return ['mean', 'max', 'min', 'std']
    
    @property
    def pna_scalars(self):
        if self.use_original_scalar_pna:
            return {'scaled', 'origin'}
        if self.use_sum_replace_pna:
            return {'scaled'}
        return {'scaled'}

    def __post_init__(self):
        assert self.n_heads_rel > 0, "n_heads_ent must be set"
        assert self.max_depth > 0, "max_depth must be set"
        assert self.embed_size > 0, "embed_size must be set"
        assert (self.in_degree_highlight >= 0. and self.in_degree_highlight <= 1.0
                or self.in_degree_highlight == -1.0),\
            "hightlight show not greater than 1.0 or less than 0. or equal -1(log)"
        if self.attn_size == 0:
            self.attn_size = self.embed_size
        if self.variant == 'DoubleAttn':
            self.ent_attn = self.ent_attn or self.rel_attn
            if self.n_heads_ent == 0:
                self.n_heads_ent = self.n_heads_rel


class RelAttnEntPnaBlock(nn.Module):

    def __init__(self, config: NEAIRConfig):
        super().__init__()
        d = config.embed_size
        h = config.attn_size
        self.embed_size = d
        self.attn_size = h
        self.in_pow = config.in_degree_highlight

        self.rel_embeds = nn.Embedding(config.n_rels, d)
        self.trans = KGEOp(config.trans_op)
        self.rel_attn = ScatterMultiHeadAttentionLayer(d, d, d, h,
                            n_heads=config.n_heads_rel, compute=config.rel_attn,
                            rel_dropout=config.rel_dropout, proj_k=True)
        self.proj_pair_R = nn.Linear(config.r_size, d)
        self.pna = PNA(d, config.pna_scalars, config.pna_aggs)
        self.res_norm = GatedResidueNorm(d, config.use_gate)

    def set_dropout(self, dropout: float):
        self.pna.set_dropout(dropout)

    def forward(self, q_rels: Tensor, state: BfsRepresents):
        G_h = state.E_r
        queries = self.rel_embeds(q_rels)
        r_q = queries.index_select(0, state.h[:,0])
        r = self.rel_embeds(state.r)
        q = G_h + r_q  
        k = r

        Rs = self.rel_attn.forward(q, k, r, state.h2rel, state.pair2rel, len(state.h2pair),
                                   unmask_aggregate=state.t2rel,
                                   unmask_shape=(len(state.t),),
                                   in_degree=state.e_i_degree).pooled_value

        Rs = self.proj_pair_R(Rs)
        G_q_h = G_h.index_select(0, state.h2pair)
        e_v = self.trans(G_q_h, Rs)
        agg_o_shape = (len(state.t), self.embed_size)
        G_t = self.pna(agg_o_shape, e_v, state.h_o_degree, state.h[:,0], q_rels.shape[0],
                       state.h2pair, state.t2pair)

        return self.res_norm(G_h, G_t, r_q, state.old_heads_len)


class SingleAttnNEAIRBlock(nn.Module):

    def __init__(self, config: NEAIRConfig):
        super().__init__()
        d = config.embed_size
        h = config.attn_size
        self.embed_size = d
        self.attn_size = h

        self.rel_embeds = nn.Embedding(config.n_rels, d)
        self.rel_attn = ScatterMultiHeadAttentionLayer(d, d, d, h, config.n_heads_rel,
                                                       compute=config.rel_attn,
                                                       proj_k=True,
                                                       rel_dropout=config.rel_dropout)
        self.proj_pair_R = nn.Linear(config.r_size, d, bias=False)
        self.trans = KGEOp(config.trans_op)
        self.proj_ent = nn.Linear(config.r_size, d, bias=False)
        self.ent_dropout = nn.Dropout(0.)
        self.res_norm = GatedResidueNorm(d, config.use_gate)

    def set_dropout(self, dropout: float):
        self.ent_dropout.p = dropout

    def forward(self, q_rels, state: BfsRepresents):
        queries = self.rel_embeds(q_rels)
        r_q = queries.index_select(0, state.h[:,0])
        G_h = state.E_r
        q = r_q + G_h
        r_k = self.rel_embeds(state.r)

        G_head = torch.index_select(G_h, 0, state.h2rel)
        G_e = self.trans(G_head, r_k)

        G_e = self.rel_attn.forward(q, r_k, G_e, state.h2rel, state.t2rel, len(state.t),
                                   unmask_aggregate=state.t2rel,
                                   unmask_shape=(len(state.t),)).pooled_value
        G_t = self.proj_ent(G_e)
        G_t = self.ent_dropout(G_t)

        return self.res_norm(G_h, G_t, r_q, state.old_heads_len)


class SinglePNANEAIRBlock(nn.Module):
    def __init__(self, config: NEAIRConfig):
        super().__init__()
        d = config.embed_size
        h = config.attn_size
        self.embed_size = d
        self.attn_size = h
        self.use_gate = config.use_gate

        self.rel_embeds = nn.Embedding(config.n_rels, d)
        self.trans = KGEOp(config.trans_op)
        self.pna = PNA(d, config.pna_scalars, config.pna_aggs)
        self.res_norm = GatedResidueNorm(d, config.use_gate)

    def set_dropout(self, dropout: float):
        self.pna.set_dropout(dropout)
    
    def forward(self, q_rels, state: BfsRepresents):
        queries = self.rel_embeds(q_rels)
        G_h = state.E_r
        r_q = queries.index_select(0, state.h[:,0])
        r_kv = self.rel_embeds(state.r)

        G_head = torch.index_select(G_h, 0, state.h2rel)
        G_e = self.trans(G_head, r_kv)
        
        G_t = self.pna(len(state.t), G_e, state.h_o_degree, state.h[:,0], q_rels.shape[0],
                       state.h2rel, state.t2rel)
        return self.res_norm(G_h, G_t, r_q, state.old_heads_len)


class NEAIR(nn.Module):

    def __init__(self, config: NEAIRConfig):
        super().__init__()
        self.config = config
        self.loader = None

        d = self.config.embed_size
        self.init_rel_embeds = nn.Embedding(self.config.n_rels, d)
        # for init its neighbors
        self.init_GNN = ScatterMultiHeadAttentionLayer(d, d, d, d, 
                                                       self.config.n_heads_rel,
                                                       compute=config.rel_attn,
                                                       proj_k=True)
        self.init_proj = nn.Linear(config.r_size, d)

        self.blks: list[RelAttnEntPnaBlock] = nn.ModuleList()
        BlkClass = {
            'SingleAttn': SingleAttnNEAIRBlock,
            'SinglePNA': SinglePNANEAIRBlock,
            'AttnPNA': RelAttnEntPnaBlock,
        }.get(config.variant)

        for _ in range(self.config.max_depth-1): # -1 depth for init at begin
            self.blks.append(BlkClass(self.config))

        self.s_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(), # nn.GELU(),
            nn.Linear(d, 1, bias=False),
            # nn.GELU()
        )

    def set_dropout(self, dropout: float):
        for blk in self.blks:
            blk.set_dropout(dropout)

    def forward(self, heads: Tensor, q_rels: Tensor) -> Tensor:
        assert self.loader is not None, "loader is not set"
        state = self.init_source_and_nei(heads, q_rels)

        # the depth of current get entities are actually i+1, means at the iteration i, 
        # we get the tail entities at detph i+1
        for i in range(1, self.config.max_depth):
            state = self.bfs_agg(i, q_rels, state)

        scores = self.score_state(q_rels, state)
        return scores
    
    def init_source_and_nei(self, heads: Tensor, q_rels: Tensor) -> BfsRepresents:
        N_b = heads.shape[0]
        h = torch.stack([torch.arange(N_b, device=heads.device), heads], dim=1)
        sel_tris = self.loader.get_neighbors(h)

        heads_with_rel_flag = torch.zeros(N_b, device=heads.device, dtype=bool)
        heads_with_rel_flag[sel_tris[:,0]] = True

        h = h[heads_with_rel_flag]            # [...coords, h_id] * B
        heads = heads[heads_with_rel_flag]    # [h_id] * B
        
        ent_pos_map = torch.full((N_b, self.loader.N_e), -1, device=heads.device, dtype=torch.long)
        neis = torch.unique(sel_tris[:, [0,3]], dim=0)

        ent_pos_map[h[:,0], h[:,1]] = torch.arange(len(h), device=heads.device)
        neis_no_repeat_mask = ent_pos_map[neis[:,0], neis[:,1]] == -1
        neis_no_repeat = neis[neis_no_repeat_mask]
        ents = torch.concat([h, neis_no_repeat], dim=0)
        ent_pos_map[neis_no_repeat[:,0], neis_no_repeat[:,1]] =\
            torch.arange(len(h), len(ents), device=heads.device)
        
        init_t_pos = ent_pos_map[sel_tris[:,0], sel_tris[:,3]]

        # [b_idx, h, r, t]
        init_q_pos = sel_tris[:,0]
        init_q = self.init_rel_embeds(q_rels)
        init_kv = self.init_rel_embeds(sel_tris[:,2])

        h2rel = ent_pos_map[sel_tris[:,0], sel_tris[:,1]]
        t2rel = init_t_pos

        # t_i_degree is highlight popular entities
        e_i_degree = torch.zeros(len(ents), device=heads.device, dtype=torch.long)
        e_i_degree.index_add_(0, t2rel, torch.ones_like(t2rel))

        E_R = self.init_GNN.forward(init_q, init_kv, init_kv, init_q_pos, init_t_pos,
                                    len(ents), unmask_aggregate=init_t_pos,
                                    unmask_shape=(len(ents),), in_degree=e_i_degree,
                                    in_pow=self.config.in_degree_highlight).pooled_value
        E_r = self.init_proj(E_R)

        if self.config.variant in {'DoubleAttn', 'AttnPNA'}:
            pairs, pair2tri = torch.unique(sel_tris[:,[0,1,3]], dim=0, return_inverse=True)
            h2pair = ent_pos_map[pairs[:,0], pairs[:,1]]
            t2pair = ent_pos_map[pairs[:,0], pairs[:,2]]
            e_o_degree = torch.zeros(len(ents), device=heads.device, dtype=torch.long)
            e_o_degree.index_add_(0, h2pair, torch.ones_like(h2pair))
        else:
            h2pair = None
            t2pair = None
            pair2tri = None
            e_o_degree = torch.zeros(len(ents), device=heads.device, dtype=torch.long)
            e_o_degree.index_add_(0, h2rel, torch.ones_like(h2rel))


        return BfsRepresents(
            ents,
            E_r,
            sel_tris[:,2],
            len(heads), 
            ent_pos_map, h2pair, t2pair, h2rel, t2rel, pair2tri,
            e_o_degree, e_i_degree)

    def bfs_agg(self, i: int, q_rels: Tensor, state: BfsRepresents) -> BfsRepresents:
        r"""
        Args:
            i: current search-depth `i`
            q_rels: shape(N_b), idx of query relations.
            last: the output of last search.
        """                
        self.collect_and_get_index(i, state)

        state.E_r = self.blks[i-1].forward(q_rels, state)
        
        return state

    def collect_and_get_index(self, i: int, state: BfsRepresents):
        r""" parse neccessary information from searched triplets 
             for rel and ent aggregation. """
        
        sel_heads = state.E[state.old_heads_len:]  # only using new ents to collect new pairs
        if sel_heads.shape[0] == 0:
            return

        sel_tris = self.loader.get_neighbors(sel_heads)
        # NOTE: we cannot get old pairs from new head nets...
        new_pairs, new_pairs2new_tris = torch.unique(sel_tris[:, [0,1,3]], dim=0, 
                                                     return_inverse=True)
        ents = torch.unique(new_pairs[:, [0,2]], dim=0)
        not_include_ent_mask = state.ent_pos_map[ents[:,0], ents[:,1]] == -1
        new_ents = ents[not_include_ent_mask]
        num_old_ents = len(state.E)

        # update_state
        state.E = torch.cat([state.E, new_ents], dim=0)
        state.ent_pos_map[new_ents[:,0], new_ents[:,1]] = torch.arange(num_old_ents, len(state.E),
                                                                       device=state.E.device)

        state.old_heads_len = num_old_ents
        new_h2rel = state.ent_pos_map[sel_tris[:,0], sel_tris[:,1]]
        state.h2rel = torch.concat([state.h2rel, new_h2rel])

        new_t2rel = state.ent_pos_map[sel_tris[:,0], sel_tris[:,3]]
        state.t2rel = torch.concat([state.t2rel, new_t2rel])

        if self.config.variant in {'DoubleAttn', 'AttnPNA'}:
            new_pairs2new_tris += len(state.h2pair)
            state.pair2rel = torch.concat([state.pair2rel, new_pairs2new_tris])
            new_h2pair = state.ent_pos_map[new_pairs[:,0], new_pairs[:,1]]
            state.h2pair = torch.concat([state.h2pair, new_h2pair])
            new_t2pair = state.ent_pos_map[new_pairs[:,0], new_pairs[:,2]]
            state.t2pair = torch.concat([state.t2pair, new_t2pair])

            new_e_o_degree = torch.zeros(len(new_ents), device=state.E.device, dtype=torch.long)
            state.e_o_degree.index_add_(0, new_h2pair, torch.ones_like(new_h2pair))
            state.e_o_degree = torch.concat([state.e_o_degree, new_e_o_degree])

            new_e_i_degree = torch.zeros(len(new_ents), device=state.E.device, dtype=torch.long)
            state.e_i_degree = torch.concat([state.e_i_degree, new_e_i_degree])
            state.e_i_degree.index_add_(0, new_t2pair, torch.ones_like(new_t2pair))
        else:
            new_e_o_degree = torch.zeros(len(new_ents), device=state.E.device, dtype=torch.long)
            state.e_o_degree.index_add_(0, new_h2rel, torch.ones_like(new_h2rel))
            state.e_o_degree = torch.concat([state.e_o_degree, new_e_o_degree])

            new_e_i_degree = torch.zeros(len(new_ents), device=state.E.device, dtype=torch.long)
            state.e_i_degree = torch.concat([state.e_i_degree, new_e_i_degree])
            state.e_i_degree.index_add_(0, new_t2rel, torch.ones_like(new_t2rel))

        state.r = torch.concat([state.r, sel_tris[:,2]])

    def score_state(self, q_rels: Tensor, state: BfsRepresents) -> Tensor:
        N_b = q_rels.shape[0]
        scores = self.s_mlp(state.E_r)
        scores.squeeze_(-1)
        scores_all = torch.zeros((N_b, self.loader.N_e), device=state.E_r.device)
        scores_all[state.E[:,0], state.E[:,1]] = scores
        
        return scores_all


@dataclass
class BfsRepresents_idd:
    E: Tensor
    E_r: Tensor
    r: Tensor
    ent_pos_map: ty.Optional[Tensor]
    h2pair: ty.Optional[Tensor]
    t2pair: ty.Optional[Tensor]
    h2rel: ty.Optional[Tensor]
    t2rel: ty.Optional[Tensor]
    pair2rel: Tensor

    h_o_degree: ty.Optional[Tensor] = None
    t_i_degree: ty.Optional[Tensor] = None

    h: Tensor = None
    old_nodes_new_idx: Tensor = None

    @property
    def t(self): 
        return self.E

    @property
    def old_heads_len(self):
        return self.old_nodes_new_idx


class NEAIR_IDD(NEAIR):

    def __init__(self, config: NEAIRConfig):
        super(NEAIR, self).__init__()
        self.config = config
        self.loader = None

        self.blks: list[RelAttnEntPnaBlock] = nn.ModuleList()
        BlkClass = {
            'SingleAttn': SingleAttnNEAIRBlock,
            'SinglePNA': SinglePNANEAIRBlock,
            'AttnPNA': RelAttnEntPnaBlock,
        }.get(config.variant, RelAttnEntPnaBlock)

        for _ in range(self.config.max_depth):
            self.blks.append(BlkClass(self.config))

        d = self.config.embed_size
        self.s_mlp = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(), #  nn.GELU(),
            nn.Linear(d, 1, bias=False),
        )

    def forward(self, heads: Tensor, q_rels: Tensor) -> Tensor:
        assert self.loader is not None, "loader is not set"
        state = self.init_source(heads, q_rels)

        # the depth of current get entities are actually i+1, means at the iteration i, 
        # we get the tail entities at detph i+1
        for i in range(0, self.config.max_depth):
            state = self.bfs_agg(i, q_rels, state)

        scores = self.score_state(q_rels, state)
        return scores

    def bfs_agg(self, i: int, q_rels: Tensor, state: BfsRepresents_idd) -> BfsRepresents_idd:
        r"""
        Args:
            i: current search-depth `i`
            q_rels: shape(N_b), idx of query relations.
            last: the output of last search.
        """                
        self.collect_and_get_index(i, state)

        # h_o_degree = self.loader.get_out_degrees(state.h[:,1])
        state.E_r = self.blks[i].forward(q_rels, state)
        
        return state

    def init_source(self, heads: Tensor, q_rels: Tensor) -> BfsRepresents_idd:
        B = heads.shape[0]
        E = torch.stack([ torch.arange(B, device=heads.device), heads ], dim=-1)
        E_r = torch.zeros((B, self.config.embed_size), device=heads.device)

        return BfsRepresents_idd(E, E_r,
                                 None, 0, None, None, None, None, None, None)

    def collect_and_get_index(self, i: int, state: BfsRepresents_idd):
        r""" parse neccessary information from searched triplets 
             for rel and ent aggregation. """
        
        sel_tris = self.loader.get_neighbors(state.E)
        state.r = sel_tris[:,2]

        if self.config.variant in {'DoubleAttn', 'AttnPNA'}:
            pairs, pair2rel = torch.unique(sel_tris[:, [0,1,3]], dim=0, 
                                           return_inverse=True, sorted=True)
            Eh, h2pair = torch.unique(pairs[:, [0,1]], dim=0, 
                                      return_inverse=True, sorted=True)
            Et, t2pair = torch.unique(pairs[:, [0,2]], dim=0, 
                                      return_inverse=True, sorted=True)
            state.h = Eh
            state.E = Et
            state.h2rel = h2pair.index_select(0, pair2rel)
            state.pair2rel = pair2rel
            state.h2pair = h2pair
            state.t2pair = t2pair
            state.h_o_degree = torch.zeros(len(Eh), device=state.E.device, dtype=torch.long)
            state.h_o_degree = state.h_o_degree.index_add_(0, h2pair, torch.ones_like(h2pair))
            
            idd_mask = state.r == self.config.n_rels-1
            self_point_pairs2rel = pair2rel[idd_mask]
            state.old_nodes_new_idx = state.t2pair.index_select(0, self_point_pairs2rel)

        else:
            Eh, h2rel = torch.unique(sel_tris[:, [0,1]], dim=0, 
                                      return_inverse=True, sorted=True)
            Et, t2rel = torch.unique(sel_tris[:, [0,3]], dim=0, 
                                      return_inverse=True, sorted=True)
            state.h = Eh
            state.E = Et
            state.h2rel = h2rel
            state.t2rel = t2rel
            idd_mask = state.r == self.config.n_rels-1
            state.old_nodes_new_idx = t2rel[idd_mask]
            
            state.h_o_degree = torch.zeros(len(Eh), device=state.E.device, dtype=torch.long)
            state.h_o_degree.index_add_(0, h2rel, torch.ones_like(h2rel))

        state.t_i_degree = torch.zeros(len(Et), device=state.E.device, dtype=torch.long)
        state.t_i_degree.index_add_(0, t2rel, torch.ones_like(t2rel))
