#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="AdaPruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -m llava.eval.model_vqa_loader \
    --model-path /path/to/checkpoint/${CKPT} \
    --question-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --image-folder ./playground/data/eval/vizwiz/test \
    --answers-file ./playground/data/eval/vizwiz/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual-token-num 576 \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --x0 14.9 \
    --k0 0.4 \
    --gamma 0.2 \
    --target-token-num ${TOKEN}

python scripts/convert_vizwiz_for_submission.py \
    --annotation-file ./playground/data/eval/vizwiz/llava_test.jsonl \
    --result-file ./playground/data/eval/vizwiz/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --result-upload-file ./playground/data/eval/vizwiz/answers_upload/${CKPT}/${METHOD}/${PARAM}.json
