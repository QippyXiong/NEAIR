
if [ "$1" = "v1" ]; then
    dataset_path="data_ind/fb237_v1"
    lr=2.5e-3
    weight_decay=3.5e-4
    dropout=0.15
    rel_dropout=0.1
    batch_size=64
    embed_size=32
    n_heads_rel=4
    max_depth=8
    use_gate=True
    num_epochs=50
    eval_batch_size=50
    attn_size=16
fi

if [ "$1" = "v2" ]; then
    dataset_path="data_ind/fb237_v2"
    lr=3e-3
    weight_decay=1e-4
    dropout=0.2
    rel_dropout=0.1
    batch_size=50
    embed_size=32
    n_heads_rel=8
    max_depth=6
    use_gate=True
    num_epochs=50
    eval_batch_size=50
    attn_size=16
fi

if [ "$1" = "v3" ]; then
    dataset_path="data_ind/fb237_v3"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.15
    rel_dropout=0.1
    batch_size=64
    embed_size=48
    n_heads_rel=4
    max_depth=5
    use_gate=True
    num_epochs=50
    eval_batch_size=50
    attn_size=16
fi

if [ "$1" = "v4" ]; then
    dataset_path="data_ind/fb237_v4"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.1
    batch_size=64
    embed_size=32
    n_heads_rel=4
    max_depth=5
    use_gate=True
    num_epochs=50
    eval_batch_size=64
    attn_size=16
fi



python neair_pna_entry.py\
    --dataset_path $dataset_path\
    --attn_size $attn_size\
    --use_gate $use_gate\
    --n_heads_rel $n_heads_rel\
    --batch_size $batch_size\
    --embed_size $embed_size\
    --num_epochs $num_epochs\
    --lr $lr\
    --dropout $dropout\
    --eval_batch_size $eval_batch_size\
    --max_depth $max_depth\
    --weight_decay $weight_decay\
    --rel_dropout $rel_dropout\
    --save_dir $CKPT_ROOT.ckpts/fb237_$1\
    "${@:2}"