#!/bin/sh

vocab="/data/vocab.json"
train_src="data/arxiv/abstract"
train_tgt="data/arxiv/title"
dev_src="data/arxiv/abstract.dev"
dev_tgt="data/arxiv/title.dev"
test_src="data/arxiv/abstract.test"
test_tgt="data/arxiv/title.test"

work_dir="work_dir"

mkdir -p ${work_dir}
echo save results to ${work_dir}

python nmt.py \
    train \
    # --cuda \
    --vocab=${vocab} \
    --train-src ${train_src} \
    --train-tgt ${train_tgt} \
    --dev-src ${dev_src} \
    --dev-tgt ${dev_tgt} \
    --input-feed \
    --valid-niter 2400 \
    --batch-size 64 \
    --hidden-size 256 \
    --embed-size 256 \
    --uniform-init 0.1 \
    --label-smoothing 0.1 \
    --dropout 0.2 \
    --clip-grad 5.0 \
    --save-to ${work_dir}/model.bin \
    --lr-decay 0.5 2>${work_dir}/err.log

python nmt.py \
    decode \
    # --cuda \
    --beam-size 5 \
    --max-decoding-time-step 100 \
    ${work_dir}/model.bin \
    ${test_src} \
    ${work_dir}/decode.txt

perl multi-bleu.perl ${test_tgt} < ${work_dir}/decode.txt >>${work_dir}/err.log
