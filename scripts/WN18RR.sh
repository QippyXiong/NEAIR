

if [ "$1" = "v1" ]; then
    dataset_path="data_ind/WN18RR_v1"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.1
    batch_size=128
    embed_size=128
    attn_size=32
    n_heads_rel=4
    max_depth=8
    num_epochs=50
    eval_batch_size=64
fi

if [ "$1" = "v2" ]; then
    dataset_path="data_ind/WN18RR_v2"
    lr=5e-3
    weight_decay=5e-4
    dropout=0.1
    rel_dropout=0.2
    batch_size=64
    embed_size=32
    attn_size=48
    n_heads_rel=8
    max_depth=5
    num_epochs=50
    eval_batch_size=64
fi

if [ "$1" = "v3" ]; then
    dataset_path="data_ind/WN18RR_v3"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.1
    batch_size=32
    embed_size=48
    attn_size=32
    n_heads_rel=4
    max_depth=6
    num_epochs=50
    eval_batch_size=64
fi

if [ "$1" = "v4" ]; then
    dataset_path="data_ind/WN18RR_v4"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.1
    batch_size=64
    embed_size=64
    attn_size=48
    n_heads_rel=4
    max_depth=5
    num_epochs=50
    eval_batch_size=64
fi


python neair_pna_entry.py\
    --rel_dropout $rel_dropout\
    --dropout $dropout\
    --batch_size $batch_size\
    --max_depth $max_depth\
    --eval_batch_size $eval_batch_size\
    --weight_decay $weight_decay\
    --lr $lr\
    --num_epochs $num_epochs\
    --n_heads_rel $n_heads_rel\
    --attn_size $attn_size\
    --embed_size $embed_size\
    --dataset_path $dataset_path\
    --save_dir $CKPT_ROOT.ckpts/WN18RR_$1\
    "${@:2}"
