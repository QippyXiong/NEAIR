import gc
import os
import logging
from dataclasses import dataclass

import torch

from common.utils import dataclass_parser, parse_dataclass_args, dataclasses_argument_parse,\
                         set_random_seed_np_torch, json_load_dataclass

from neair import NEAIR, NEAIRConfig, NEAIR_IDD
from NEAIRTrainer import NEAIRTrainer, NEAIRControlParams, NEAIRTrainParams
from loader import InducKGDataset, InducKGDatasetConfig, KGDataset


KGDataset.create_in_out_degree = True
NEAIRTrainer.HyperparamsCls = NEAIRConfig
NEAIRTrainer.ModelCls = NEAIR


@dataclass
class MoreArgs:
    resume_path: str = None
    device: str      = 'cuda'
    only_test: bool  = False
    test_while_valid: bool = False
    save_dir: str =  '.neair_pna'
    resume_epoch: int = None
    resume_step: int = None
    log_level: int = logging.INFO  # logging.ERROR
    random_seed: int = 1234
    use_optimized_model: bool = False

def main():

    parser = dataclass_parser(MoreArgs)
    args, unkown = parser.parse_known_args()
    margs = parse_dataclass_args(args, MoreArgs)
    set_random_seed_np_torch(margs.random_seed)

    if margs.resume_path:
        dsc = json_load_dataclass(os.path.join(margs.resume_path, 'dataset.json'),
                                  InducKGDatasetConfig)
        dsc.pt_device = margs.device
        ds = InducKGDataset(dsc)
        if dsc.add_idd:
            NEAIRTrainer.ModelCls = NEAIR_IDD
        trainer = NEAIRTrainer.from_checkpoint(margs.resume_path, dataset=ds,
                                               epoch=margs.resume_epoch,
                                               step=margs.resume_step,
                                               use_optimized_model=margs.use_optimized_model)
    else:
        hps, tps, cps, dsc = dataclasses_argument_parse(
            NEAIRConfig, NEAIRTrainParams, NEAIRControlParams, InducKGDatasetConfig,
            args_seq=unkown
        )
        assert hps.max_depth != -1, "should give a valid max_depth selection."
        if torch.__version__ >= (2,5,0) and margs.use_optimized_model:
            torch.compiler.allow_in_graph(torch.sparse_coo_tensor)
        dsc.pt_device = margs.device
        dataset = InducKGDataset(dsc)
        hps.n_rels = dataset.dataset.N_r
        
        model = NEAIR(hps) if not dsc.add_idd else NEAIR_IDD(hps)
        trainer = NEAIRTrainer(model, hps, tps, cps, margs.save_dir, dataset,
                              test_while_valid=margs.test_while_valid,
                              use_optimized_model=margs.use_optimized_model)
    trainer.set_logger_level(margs.log_level)
    trainer.to(margs.device)

    test_metrics = (trainer.test() if margs.only_test
                    else trainer.train())
    
    trainer.logger.info(f'save_dir: {trainer.save_dir}')
    trainer.dataset = None
    gc.collect()

    return test_metrics, trainer.best_valid_metrics, str(trainer.save_dir)


if __name__ == '__main__':
    test_metrics, best_valid_metrics, save_dir = main()
    print('test   ', test_metrics)
    print('valid  ', best_valid_metrics)
    print('save_to', save_dir)
