#!/bin/bash

CKPT="llava-v1.6-vicuna-7b"
METHOD="AdaPruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path ./models/llava-v1.6-vicuna-7b/ \
    --question-file ./playground/data/eval/MME/llava_mme.jsonl \
    --image-folder ./playground/data/eval/MME/MME_Benchmark_release_version \
    --answers-file ./playground/data/eval/MME/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual-token-num 2880 \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --x0 14.9 \
    --k0 0.4 \
    --gamma 0.2 \
    --target-token-num ${TOKEN}

cd ./playground/data/eval/MME

python convert_answer_to_mme.py --experiment ${CKPT}/${METHOD}/${PARAM}

cd eval_tool

python calculation.py --results_dir answers/${CKPT}/${METHOD}/${PARAM}
