#!/bin/bash

module load anaconda/3
conda activate fairseq

src=$1
tgt=$2
type=$3
mode=$4
layers=$5
embdim=$6
ffndim=$7
heads=$8

base=$mode"_"$layers"_"$embdim"_"$ffndim"_"$heads

if [[ $type == "Stacked" ]]; then
  base="Stacked_"$base
  arch="transformer"
elif [[ $type == "Universal" ]]; then
  base="Universal_"$base
  arch="universal_transformer"
fi

base="wmt14_"$src"-"$tgt"/"$base

mkdir -p "logs/wmt14_"$src"-"$tgt"/"

name="/miniscratch/mittalsa/NMT/checkpoints/"$base
tb="/miniscratch/mittalsa/NMT/tensorboard/"$base
base="logs/"$base
wandb=$src"-"$tgt

echo Running name is $name

python3 fairseq_cli/train.py \
    "/miniscratch/mittalsa/NMT/data-bin/wmt14_"$src"_"$tgt \
    --tensorboard-logdir $tb \
    --wandb-project $wandb \
    --arch $arch --share-decoder-input-output-embed \
    --encoder-embed-dim	$embdim \
    --encoder-ffn-embed-dim $ffndim \
    --decoder-embed-dim	$embdim \
    --decoder-ffn-embed-dim $ffndim \
    --encoder-layers $layers \
    --decoder-layers $layers \
    --encoder-attention-heads $heads \
    --decoder-attention-heads $heads \
    --attention-rules $rules \
    --attention-type $mode \
    $extras \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 7.5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.1 --weight-decay 0.0 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 8 \
    --max-epoch	100 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --save-dir $name > $base".log"
