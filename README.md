# NEAIR: Neighbor-Aware Inductive Reasoning for Knowledge Graph Completion

## requirements

We used python==3.11 for experiments, but python >= 3.9 should be okay for these code.

Package Requirements:

```
torch >= 2.0.1
scipy
```

## run

Datasets are required for preprocessing before experiments:

```bash
python preprocess.py -d data_ind/fb237_v1 -i
python preprocess.py -d data_ind/WN18RR_v1 -i
python preprocess.py -d data_ind/nell_v1 -i
python preprocess.py -d data/WN18RR
python preprocess.py -d data/nell

# run for all datasets
bash scripts/preprocess_all.sh
```

Simply run following commands to reproduce our experiment results:

```bash
# for inductive 
bash scripts/FB15k237.sh v1 --device 'cuda'

# for transductive
bash scripts/tran.sh WN18RR --device 'cuda'

# different transition function for message passing DistMult/TransE/RotatE
bash scripts/FB15k237.sh v1 --device 'cuda' --trans_op RotatE

# using optimized model in pytorch is highly recommended for transductive setting
# but should install torch >= 2.5.0
bash scripts/tran.sh WN18RR --device 'cuda' --use_optimized_model True
```
