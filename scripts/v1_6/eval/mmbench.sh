#!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CKPT="llava-v1.6-vicuna-7b"
METHOD="AdaPruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_mmbench \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --single-pred-prompt \
    --visual-token-num 2880 \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --x0 14.9 \
    --k0 0.4 \
    --gamma 0.2 \
    --target-token-num ${TOKEN}

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/mmbench_dev_20230712.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/${CKPT}/${METHOD} \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/${CKPT}/${METHOD} \
    --experiment ${PARAM}
