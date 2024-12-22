
if [ "$1" = "WN18RR" ]; then
    dataset_path="data/WN18RR"
    induc="False"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.1
    rel_dropout=0.1
    max_depth=6
    batch_size=64
    embed_size=48
    n_heads_rel=4
    num_epochs=50
    save_span_epoch=1
    valid_span_epoch=5
    eval_batch_size=64
    attn_size=16 # 32
    valid_after_train=True
    use_optimized_model=True
fi

if [ "$1" = "NELL995" ]; then
    dataset_path="data/nell"
    induc="False"
    lr=1e-3
    weight_decay=1e-4
    dropout=0.15
    rel_dropout=0.1
    max_depth=5
    batch_size=32
    embed_size=36
    n_heads_rel=3
    num_epochs=50
    save_span_epoch=1
    valid_span_epoch=5
    eval_batch_size=16
    attn_size=12
    valid_after_train=True
    use_optimized_model=True
fi


# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 
python neair_pna_entry.py\
    --lr $lr\
    --max_depth $max_depth\
    --embed_size $embed_size\
    --valid_span_epoch $valid_span_epoch\
    --induc $induc\
    --eval_batch_size $eval_batch_size\
    --dataset_path $dataset_path\
    --rel_dropout $rel_dropout\
    --n_heads_rel $n_heads_rel\
    --save_span_epoch $save_span_epoch\
    --num_epochs $num_epochs\
    --weight_decay $weight_decay\
    --batch_size $batch_size\
    --dropout $dropout\
    --attn_size $attn_size\
    --save_dir $CKPT_ROOT.ckpts/$1\
    --valid_after_train $valid_after_train\
    --use_optimized_model $use_optimized_model\
    ${@:2}
