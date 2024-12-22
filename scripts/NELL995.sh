
if [ "$1" = "v1" ]; then
    lr=5e-3
    max_depth=4
    dataset_path="data_ind/nell_v1"
    embed_size=32
    weight_decay=1e-4
    dropout=0.2
    rel_dropout=0.1
    n_heads_rel=4
    batch_size=64
    eval_batch_size=64
    attn_size=32
fi

if [ "$1" = "v2" ]; then
    lr=5e-3
    max_depth=3
    dataset_path="data_ind/nell_v2"
    embed_size=32
    use_gate=True
    weight_decay=1e-4
    dropout=0.2
    rel_dropout=0.1
    n_heads_rel=4
    batch_size=80
    eval_batch_size=32
    attn_size=32
fi

if [ "$1" = "v3" ]; then
    lr=5e-3
    max_depth=4
    dataset_path="data_ind/nell_v3"
    embed_size=32
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.2
    n_heads_rel=4
    batch_size=64
    eval_batch_size=32
    attn_size=32
fi

if [ "$1" = "v4" ]; then
    lr=1e-3
    max_depth=6
    dataset_path="data_ind/nell_v4"
    embed_size=64
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.2
    n_heads_rel=4
    batch_size=32
    eval_batch_size=64
    attn_size=32
fi



python neair_pna_entry.py\
    --embed_size $embed_size\
    --dropout $dropout\
    --n_heads_rel $n_heads_rel\
    --max_depth $max_depth\
    --lr $lr\
    --eval_batch_size $eval_batch_size\
    --dataset_path $dataset_path\
    --batch_size $batch_size\
    --attn_size $attn_size\
    --weight_decay $weight_decay\
    --rel_dropout $rel_dropout\
    --save_dir $CKPT_ROOT.ckpts/nell_$1\
    "${@:2}"