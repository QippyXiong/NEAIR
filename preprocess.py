import os
import json
import pickle

import numpy as npy


def preprocess_triplets_file(entity2id: dict, relation2id: dict, file_path) -> npy.ndarray:

    assert isinstance(entity2id, dict) and isinstance(relation2id, dict),\
        "should pass dict entity2id and relation2id."
    
    fp = open(file_path, 'r', encoding='UTF-8')

    tris = []
    for line in fp:
        head, rel, tail = line.strip().split()
        h_id = entity2id.setdefault(head, len(entity2id))
        t_id = entity2id.setdefault(tail, len(entity2id))
        r_id = relation2id.setdefault(rel, len(relation2id))
        tris.append([h_id, r_id, t_id])

    fp.close()
    return npy.array(tris)


def build_tris_label(tris, head_ans_dict, N_e, build_filter = False) -> npy.ndarray:    
    tris_label = npy.zeros((len(tris), N_e), dtype=bool)
    for i, (h, r, t) in enumerate(tris):
        ans = head_ans_dict[(h, r)]
        tris_label[i, ans] = True
        if build_filter:
            tris_label[i, t] = False
    return tris_label


def preprocess_kg_dataset(dataset_dir, entity2id = None, relation2id = None, train_include_valid = False):
    r"""
    dataset should be in this structure:
    - dataset_dir
      - train.txt
      - valid.txt
      - test.txt
    After preprocess it would be like:
    - dataset_dir
      - train.txt
      - valid.txt
      - test.txt
      - entity.txt
      - relation.txt
      - tris.npy       [ train-valid-test (int64)]
      - tris_label.npy [ train_label-valid_filter-test_filter (bool)]
      - dataset_info.json
    NOTICE: train, valid, test all includes the inverse
    """
    entity2id = entity2id if entity2id is not None else {}
    relation2id = relation2id if relation2id is not None else {}
    tris = {}
    dataset_info = {}
    for split in ['train', 'valid', 'test']:
        file_path = os.path.join(dataset_dir, f"{split}.txt")
        split_npy = preprocess_triplets_file(entity2id, relation2id, file_path)
        tris[split] = split_npy
        dataset_info[split] = split_npy.shape[0]

    dataset_info['n_entities'] = len(entity2id)
    dataset_info['n_relations'] = len(relation2id)
    N_e, N_r = len(entity2id), len(relation2id)

    for split in ['train', 'valid', 'test']:
        split_npy = tris[split]
        inv_split_npy = split_npy[:, [2,1,0]].copy()
        inv_split_npy[:, 1] += N_r
        tris['inv_'+split] = inv_split_npy

    head_ans_dict = {}
    for split in ['train', 'inv_train']:
        for h, r, t in tris[split]:  # train_label only including train ans
            head_ans_dict.setdefault((h, r), []).append(t)

    if not train_include_valid:  # in inductive setting, will include valid for training
                                 # and use test for validation for training graph
        with open(os.path.join(dataset_dir, 'train_label.pkl'), 'wb') as fp:
            pickle.dump(head_ans_dict, fp)

    # bias includes all the triplets answers
    for split in ['valid', 'inv_valid']:
        for h, r, t in tris[split]:
            head_ans_dict.setdefault((h, r), []).append(t)
    
    if train_include_valid:
        with open(os.path.join(dataset_dir, 'train_label.pkl'), 'wb') as fp:
            pickle.dump(head_ans_dict, fp)

    for split in ['test', 'inv_test']:
        for h, r, t in tris[split]:
            head_ans_dict.setdefault((h, r), []).append(t)

    with open(os.path.join(dataset_dir, 'eval_bias.pkl'), 'wb') as fp:
        pickle.dump(head_ans_dict, fp)
    
    splits = ['train', 'valid', 'test', 'inv_train', 'inv_valid', 'inv_test' ]
    final_tris = npy.concatenate([ tris[split] for split in splits ])
    print(f"{final_tris.shape=}")
    npy.save(os.path.join(dataset_dir, 'tris.npy'), final_tris)

    with open(os.path.join(dataset_dir, 'entity.txt'), 'w', encoding='UTF-8') as fp:
        for ent, ent_id in entity2id.items():
            fp.write(f"{ent}\t{ent_id}\n")

    with open(os.path.join(dataset_dir, 'relation.txt'), 'w', encoding='UTF-8') as fp:
        for rel, rel_id in relation2id.items():
            fp.write(f"{rel}\t{rel_id}\n")

    with open(os.path.join(dataset_dir, 'dataset_info.json'), 'w', 
              encoding='utf-8') as fp:
        json.dump(dataset_info, fp)


def preprocess_kg_dataset_ind(dataset_dir):
    entity2id, relation2id = {}, {}
    preprocess_kg_dataset(dataset_dir, entity2id, relation2id, train_include_valid=True)
    ori_rels = relation2id.copy()
    preprocess_kg_dataset(str(dataset_dir)+'_ind', relation2id=relation2id)
    assert len(ori_rels) == len(relation2id),\
        "Relation types of two inductive datasets should be same."
    for key, ori_v in ori_rels.items():
        assert ori_v == relation2id[key],\
            "Relation types of two inductive datasets should be same."


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-d', '--dir', help='Knowledge graph dataset directory.')
    parser.add_argument('-ind', '--is-inductive', action='store_true',
                        help='If the knowledge graph is inductive reasoning.')

    args = parser.parse_args()
    
    if args.is_inductive:
        preprocess_kg_dataset_ind(args.dir)
    else:
        preprocess_kg_dataset(args.dir)
