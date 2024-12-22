import gc
import os
import pickle
import typing as ty
from pathlib import Path
from dataclasses import dataclass, asdict
from copy import deepcopy

import torch
import numpy as np
from scipy.sparse import csr_matrix
from torch import Tensor
import torch.utils
import torch.utils.data

from common.interfaces import DatasetConfig
from common import utils


@dataclass(unsafe_hash=True)
class KGDatasetConfig(DatasetConfig):
    dataset_path: str = 'data/WN18RR'
    inv_rel: bool = True
    fact_proportion: float = 0.8
    add_idd: bool = False
    pt_device: str = ''
    induc: bool = False
    test_only: bool = False


@dataclass
class KGInfo:
    n_entities: int
    n_relations: int
    train: int
    valid: int
    test: int


class KGDataset:

    def __init__(self, config: KGDatasetConfig) -> None:
        self.config = config
        dataset_dir = Path(config.dataset_path)
        dataset_info = utils.json_load_dataclass(dataset_dir/'dataset_info.json',
                                                 KGInfo)
        self.tris = np.load(dataset_dir/'tris.npy')
        self.ent2id = utils.read_name_idx_txt(dataset_dir/'entity.txt')
        self.rel2id = utils.read_name_idx_txt(dataset_dir/'relation.txt')
        self.L = self.tris.shape[0] // 2  
        # tris is composed of [ train, valid, test, inv_train, inv_valid, inv_test ]

        split_begins = { 'train': 0, 'valid': dataset_info.train,
                         'test': dataset_info.train + dataset_info.valid }
        split_lens = { 'train': dataset_info.train, 'valid': dataset_info.valid,
                       'test': dataset_info.test }
        
        self.split_begins = split_begins
        self.split_lens = split_lens

        self.M_nei = None
        self.pt_facts = None
        
        if self.config.inv_rel:
            n_r = len(self.rel2id)
            inv_rel2id = {}
            for k, v in self.rel2id.items():
                inv_rel2id[f"inv_{k}"] = v+n_r
            self.rel2id.update(inv_rel2id)

        if self.config.add_idd:  # add idd_tris for self.tris
            assert self.config.inv_rel, "Might wrong for embedding models adding idd rels."
            ents = np.array(list(self.ent2id.values()))
            if 'idd' not in self.rel2id:
                self.rel2id['idd'] = len(self.rel2id)
            idds = np.full_like(ents, self.rel2id['idd'], dtype=np.longlong)
            idd_tris = np.stack([ents, idds, ents], axis=-1)
            self.tris = np.concatenate([self.tris, idd_tris])

        # this if-else dealing with train_index and facts_index,
        # while changing the begin and len of valid and test
        if self.config.induc:
            assert self.config.inv_rel, "Inductive model uses the inverse rel assumption"

            if self.config.test_only:
                # full_facts, no_train
                self.facts_index = np.arange(split_begins['train'],
                                             split_begins['train']+split_lens['train'])
                inv_facts_index = self.facts_index + self.L
                self.facts_index = np.concatenate([self.facts_index, inv_facts_index])
                
                self.train_index = np.array([], dtype=np.int64)
                # for test_only inductive dataset, move valid as test
                split_lens['test'] += split_lens['valid']
                split_begins['test'] = split_begins['valid']
                split_lens['valid'] = 0
                self.maybe_add_idd_facts()
                self.build_nei_mapper()

            else:
                # for training inductive dataset, merge valid to train, move test as valid
                split_lens['train'] += split_lens['valid']
                split_lens['valid'] = split_lens['test']
                split_begins['valid'] = split_begins['test']
                split_lens['test'] = 0
                self.split_facts_train()

        else:
            if 1.0 > self.config.fact_proportion > 0.:
                self.split_facts_train()

            else:  # this situation no facts, all for train
                self.facts_index = np.array([], dtype=np.int64)
                self.train_index = np.arange(split_begins['train'],
                                             split_begins['train']+split_lens['train'])

        self.valid_index = np.arange(split_begins['valid'],
                                     split_begins['valid']+split_lens['valid'])
        self.test_index = np.arange(split_begins['test'],
                                    split_begins['test']+split_lens['test'])
        if self.config.inv_rel:
            inv_valid_index = self.valid_index + self.L
            self.valid_index = np.concatenate([self.valid_index, inv_valid_index])
            inv_test_index = self.test_index + self.L
            self.test_index = np.concatenate([self.test_index, inv_test_index])

        with open(dataset_dir/'train_label.pkl', 'rb') as fp:
            self.train_label = pickle.load(fp)
        
        with open(dataset_dir/'eval_bias.pkl', 'rb') as fp:
            self.filter_bias = pickle.load(fp)

        self.history_facts_train_index = None

    def maybe_add_idd_facts(self):
        if self.config.add_idd:
            assert self.tris.shape[0] - self.L*2 == len(self.ent2id),\
                "Should have the idd tris add into original tris but got "\
                f"{self.L=}, {len(self.ent2id)=}, {len(self.tris)=}"
            if isinstance(self.facts_index, slice):
                ori_facts_index = np.arange(self.facts_index.start, self.facts_index.stop)
            elif isinstance(self.facts_index, np.ndarray):
                ori_facts_index = self.facts_index
            else:
                raise TypeError(f"Unexpected facts index type "
                                f"{type(self.facts_index).__name__}")

            idd_facts_index = np.arange(self.L*2, len(self.tris))
            self.facts_index = np.concatenate([ori_facts_index, idd_facts_index])

    def split_facts_train(self, restore: bool = False):
        r""" this faction will call `maybe_add_idd_facts` and `build_nei_mapper` """
        assert 1.0 > self.config.fact_proportion > 0.,\
            "should have a reasonable fact proportion"

        if restore:
            assert self.history_facts_train_index is not None,\
                "Cannot restore history facts_index and train_index"
            self.facts_index, self.train_index =\
                self.history_facts_train_index
            self.build_nei_mapper()
            return

        L_t = self.split_lens['train']
        full_train_tris_index = np.random.permutation(\
            np.arange(self.split_begins['train'], 
                    self.split_begins['train']+L_t))

        L_f = int(L_t * self.config.fact_proportion)
        ori_facts_index = full_train_tris_index[:L_f]
        inv_facts_index = ori_facts_index + self.L
        ori_train_index = full_train_tris_index[L_f:]
        inv_train_index = ori_train_index + self.L
        self.facts_index = np.concatenate([ori_facts_index, inv_facts_index])
        self.train_index = np.concatenate([ori_train_index, inv_train_index])

        # assert not (~(self.facts[:L_f,0] == self.facts[L_f:,2])).any()
        # exculde ents those without any relation
        # ents_with_rel_flag = np.zeros(self.N_e, dtype=bool)
        # ents_with_rel_flag[self.facts[:,0]] = True
        # train_heads = self.train[:, 0]
        # train_index_flag = ents_with_rel_flag[train_heads]
        # self.train_index = self.train_index[train_index_flag]

        self.maybe_add_idd_facts()
        self.build_nei_mapper()

    def merge_facts(self):
        r""" this faction will call `maybe_add_idd_facts` and `build_nei_mapper` """
        if len(self.train_index) == 0:
            return  # already merged
        self.history_facts_train_index = self.facts_index, self.train_index

        full_train_tris_index = np.arange(self.split_begins['train'],
                                          self.split_begins['train']+self.split_lens['train'])
        self.facts_index = np.concatenate([ full_train_tris_index, full_train_tris_index+self.L ])
        self.train_index = np.array([], dtype=np.int64)
        self.maybe_add_idd_facts()
        self.build_nei_mapper()

    facts     = property(lambda self: self.tris[self.facts_index])
    train     = property(lambda self: self.tris[self.train_index])
    valid     = property(lambda self: self.tris[self.valid_index])
    test      = property(lambda self: self.tris[self.test_index])

    N_f = property(lambda self: (self.facts_index.stop-self.facts_index.start
                                 if isinstance(self.facts_index, slice) else
                                 len(self.facts_index)))
    N_e = property(lambda self: len(self.ent2id))
    N_r = property(lambda self: len(self.rel2id))

    def build_nei_mapper(self):
        if self.config.pt_device:
            self.build_nei_mapper_pt()
        else:
            self.build_nei_mapper_np()

    def build_nei_mapper_np(self, _ = None):
        r""" 
        Build M_nei for get neighbors, if add_idd will add idd triplets in facts.
        Ensure self.facts has been set before calling this method.
        """
        print("build_nei_mapper_np called")
        self.M_nei = csr_matrix(
            (np.ones(self.N_f), (self.facts[:, 0], np.arange(self.N_f))), 
            shape=(self.N_e, self.N_f), dtype=bool)

    def build_nei_mapper_pt(self):
        N_t, N_e = self.N_f, self.N_e
        device = torch.device(self.config.pt_device)
        heads = torch.from_numpy(self.facts[:, 0])
        col_idxes = torch.arange(N_t)
        idxes = torch.stack((heads, col_idxes), dim=0)

        self.M_nei = torch.sparse_coo_tensor(idxes, torch.ones(N_t),
                                             (N_e, N_t)).to(device)
        del idxes, col_idxes
        del heads
        self.pt_facts = torch.from_numpy(self.facts).to(device)
        gc.collect()
        # torch.cuda.empty_cache()

    def get_neighbors(self, heads: Tensor):
        assert self.M_nei is not None, "Please call build_nei_mapper before calling this method."
        return (self.get_neighbors_pt(heads) if self.config.pt_device else
                self.get_neighbors_np(heads))

    def get_neighbors_np(self, heads: Tensor):
        r"""   
        Args:
            `heads`: node_idx => (..., h_id), and should be sorted
            
        Shapes:
            `old_map`: (P, 3)
        
        Returns:
            `new_map`: (P', 3), path_id => (batch_idx, r_id, t_id)
        """
        device = 'cpu'
        if isinstance(heads, np.ndarray):
            map_np = heads
        elif isinstance(heads, Tensor):
            map_np = heads.cpu().numpy()
            device = heads.device
        else:
            raise ValueError(f"old_map should be numpy.ndarray or torch.Tensor "
                             f"but got {type(heads).__name__}")

        new_p, nei_tris = self.M_nei[map_np[:, -1]].nonzero()
        sel_tris = self.facts[nei_tris, :]  # 
        coord = torch.from_numpy(map_np[new_p, :-1]).to(device)
        sel_tris = torch.from_numpy(sel_tris).to(device)

        if sel_tris.dim() == 1:  # in some cases might only select one triplet ...
            assert coord.shape[0] == 1, "Error for batch_idxes and sel_tris."
            sel_tris = sel_tris.unsqueeze(0)

        sel_tris = torch.concat((coord, sel_tris), dim=1)

        return sel_tris

    def get_neighbors_pt(self, heads: Tensor):
        r"""
        Args:
            `heads`: [..., e_idx] x N
        """
        # tri_idx_one_hop = one_hot_heads @ self.M_nei
        idxes = torch.cat([torch.arange(len(heads), device=heads.device).unsqueeze(0),
                           heads[:, -1].unsqueeze(0)], dim=0)
        one_hot_heads = torch.sparse_coo_tensor(idxes, torch.ones(len(heads), device=heads.device), 
                                                (len(heads), self.N_e), device=heads.device)
        sel_idx = torch.mm(one_hot_heads, self.M_nei).indices()
        
        new_p, new_tris = sel_idx[0, :], sel_idx[1, :]
        
        sel_tris = self.pt_facts[new_tris]
        coords =  heads[new_p, :-1]

        if sel_tris.dim() == 1:  # in some cases might only select one triplet ...
            assert coords.shape[0] == 1, "Error for batch_idxes and sel_tris."
            sel_tris = sel_tris.unsqueeze(0)

        sel_tris = torch.concat((coords, sel_tris), dim=1)

        return sel_tris
    
    class KGEvalDataset:

        def __init__(self, tris, filter_bias, N_e):
            self.tris = tris
            self.filter_bias = filter_bias
            self.N_e = N_e
        
        def __len__(self):
            return len(self.tris)

        def __getitem__(self, idx):
            bias = np.zeros(self.N_e, dtype=bool)
            h,r,t = self.tris[idx]
            bias[self.filter_bias[(h,r)]] = True
            bias[t] = False
            return self.tris[idx], bias

    def get_evaluate_dataset(self, is_valid: bool):
        return (self.KGEvalDataset(self.valid, self.filter_bias, self.N_e) if is_valid else
                self.KGEvalDataset(self.test, self.filter_bias, self.N_e))

    def __str__(self) -> str:
        dataset_name = os.path.basename(self.config.dataset_path)
        name = f'{type(self).__name__}({dataset_name})'
        name += '_inv' if self.config.inv_rel else ''
        return name
    
    def __len__(self) -> int:
        if self.train is None:
            raise ValueError("You may have merged train into facts but run it for "
                             "training or forgot to re-split, check if such problem exists.")
        return len(self.train_index)

    def __getitem__(self, idx: int) -> np.ndarray:
        tris_idx = self.train_index[idx]
        label = np.zeros(self.N_e, dtype=bool)
        triplet = self.tris[tris_idx]
        h,r,_ = triplet
        if (h,r) not in self.train_label:
            print("triplet:", triplet)
        label[self.train_label[(h,r)]] = True
        return triplet, label

    # @override
    def dump_config(self) -> dict:
        r""" dump dataset basic information or setting into json serializable dict """
        return asdict(self.config)

    # @override
    @classmethod
    def from_config(cls, config: dict):
        dataset_config = KGDatasetConfig(**config)
        return cls(dataset_config)


@dataclass(unsafe_hash=True)
class InducKGDatasetConfig(KGDatasetConfig):
    dataset_path: str = 'data_ind/fb15k237_v1'
    inv_rel: bool = True
    fact_proportion: float = 0.8
    add_idd: bool = False
    pt_device: str = ''
    induc: bool = True
    test_only: bool = False

    def __post_init__(self):
        if self.induc and not self.inv_rel:
            raise ValueError("Induction may only support inversed relation, "
                             "any mistakes?")


class InducKGDataset:

    def __init__(self, config: InducKGDatasetConfig):
        self.config = config
        
        if config.induc:
            if config.test_only:
                raise ValueError("test_only should not be set as True by hand.")
            dataset = KGDataset(config)
            ind_dsc = deepcopy(config)
            ind_dsc.dataset_path = str(config.dataset_path) + '_ind'
            ind_dsc.test_only = True
            ind_dsc.fact_proportion = 1.0
            ind_dataset = KGDataset(ind_dsc)
            self.dataset = dataset
            self.ind_dataset = ind_dataset
        else:
            self.dataset = KGDataset(config)

    # @override
    def dump_config(self) -> dict:
        r""" dump dataset basic information or setting into json serializable dict """
        return asdict(self.config)

    # @override
    @classmethod
    def from_config(cls, config: dict):
        dataset_config = utils.dict_dump_dataclass(InducKGDatasetConfig, config)
        return cls(dataset_config)
    
    def __getattr__(self, name: str):
        return getattr(self.dataset, name)


def test_ind_dataset_surjective(ds, B):
    # test get_neighbors function
    for dataset in [ds.dataset, ds.ind_dataset]:
        ents = np.arange(dataset.N_e)
        ents_with_rel_flag = np.zeros_like(ents)
        ents_with_rel_flag[dataset.facts[:,0]] = True

        ents_with_least_one_rel = ents[ents_with_rel_flag]
        for i in range(0, dataset.N_e, B):
            head_ents = torch.from_numpy(ents_with_least_one_rel[i:i+B])
            # in 10 search times, no ents dropped
            heads = torch.stack([torch.arange(head_ents.shape[0]), head_ents], dim=-1)

            if pt_device:
                heads = heads.to(pt_device)

            if not add_idd:
                neis = dataset.get_neighbors(heads)[:,[0,3]]
                heads = torch.unique(torch.concat([heads, neis]), dim=0)

            buffer = torch.empty((B, dataset.N_e), dtype=bool, device=heads.device)
            for _ in range(10):
                sel_tris = dataset.get_neighbors(heads)
                new_heads = torch.unique(sel_tris[:,[0,3]], dim=0)

                buffer.fill_(True)
                buffer[new_heads[:,0], new_heads[:,1]] = False
                include_old_flags = buffer[heads[:,0], heads[:,1]]
                assert not include_old_flags.any(), "Should includes all the old entities"\
                                                    f"but {heads[include_old_flags].tolist()}"\
                                                    "not in new set."
                heads = new_heads


@dataclass
class SearchResults:
    h: torch.Tensor
    t: torch.Tensor
    tris: torch.Tensor


def idd_search_algorithm(loader: KGDataset, last: SearchResults):
    sel_tris = loader.get_neighbors(last.t)
    Eh = torch.unique(sel_tris[:, [0,1]], dim=0, sorted=True)
    Et = torch.unique(sel_tris[:, [0,3]], dim=0, sorted=True)

    return SearchResults(h=Eh, t=Et, tris=sel_tris)


def near_search_algorithm(loader: KGDataset, last: SearchResults):
    delta_E = last.t[len(last.h):]
    new_tris = loader.get_neighbors(delta_E)
    new_ents = torch.unique(new_tris[:,[0,3]], dim=0)
    h = last.t
    t = torch.cat([last.h, new_ents], dim=0)
    tris = torch.cat([last.tris, new_tris], dim=0)
    return SearchResults(h=h, t=t, tris=tris)

TAlgo = ty.Callable[[KGDataset, SearchResults], SearchResults]
AlgoDict: dict[ty.Literal['near', 'idd'], TAlgo] = {
    'near': near_search_algorithm,
    'idd': idd_search_algorithm
}


def export_search_results_subgraph(loader: KGDataset, h, max_depth, algo):
    heads = torch.tensor([ [0, h] ]).to(loader.config.pt_device)

    method = AlgoDict[algo]

    results = []
    empty_h = torch.empty([0,2], dtype=torch.long, device=loader.config.pt_device)
    empty_tris = torch.empty([0,4], dtype=torch.long, device=loader.config.pt_device)
    state = SearchResults(empty_h, heads, empty_tris)

    id2ent = { k: v for v, k in loader.ent2id.items() }
    id2rel = { k: v for v, k in loader.rel2id.items() }

    def convert_tris_to_str_subgraph(tris: Tensor) -> list[tuple[str]]:
        list_tris = tris.tolist()
        subgraph = [
            (id2ent[h], id2rel[r], id2ent[t])
            for _, h, r, t in list_tris
        ]
        return subgraph

    results = []
    for i in range(max_depth):
        state = method(loader, state)
        subgraph = convert_tris_to_str_subgraph(state.tris)
        results.append(subgraph)
        print(f"{i+1}th depth: {len(subgraph)} triples")
    
    return results


def main():
    @dataclass
    class LoaderTask:
        task: ty.Literal['Runtime', 'Subgraph', 'Surjective']
        algo: ty.Literal['near', 'idd'] = 'near'
        h_ent_idx: int = 0
        max_depth: int = 3
        batch_size: int = 32

    config, info = utils.dataclasses_argument_parse(KGDatasetConfig, LoaderTask)
    
    if info.task == 'Subgraph':
        loader = KGDataset(config)
        loader.merge_facts()
        results = export_search_results_subgraph(loader, info.h_ent_idx,
                                                info.max_depth, info.algo)
        for i, subgraph in enumerate(results):
            print(f"{i+1}th depth:")
            for triple in subgraph:
                print(triple)
    
    elif info.task == 'Runtime':
        loader = KGDataset(config)
        train_loader = torch.utils.data.DataLoader(loader, info.batch_size)
        num_samples = len(train_loader)
        method = AlgoDict[info.algo]
        Coverage = 0.
        with utils.TimeRecoder(f"Runtime For Single Epoch of {loader}"):
            for batch in train_loader:
                if loader.config.pt_device:
                    batch = batch.to(loader.config.pt_device)
                ents = batch[:,0]
                empty_h = torch.empty([0,2], dtype=torch.long, device=loader.config.pt_device)
                empty_tris = torch.empty([0,4], dtype=torch.long, device=loader.config.pt_device)
                state = SearchResults(empty_h, ents, empty_tris)
                for i in range(1, info.max_depth):
                    state = method(loader, state)
                Coverage += len(state.t)/loader.N_e
        Coverage = Coverage/num_samples
        print(f"average coverage: {Coverage:.4f}")

    elif info.task == 'Surjective':
        config = InducKGDatasetConfig(**asdict(config))
        ds = InducKGDataset(config)
        test_ind_dataset_surjective(ds, info.batch_size)

    else:
        raise ValueError(f"Unkown task {info.task}")


if __name__ == '__main__':
    main()
