#!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

CKPT="llava-v1.6-vicuna-7b"
METHOD="AdaPruner"
TOKEN=${1}
PARAM="n_${TOKEN}"

python -W ignore -m llava.eval.model_vqa_loader \
    --model-path ./models/llava-v1.6-vicuna-7b/ \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
    --visual-token-num 2880 \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --x0 14.9 \
    --k0 0.4 \
    --gamma 0.2 \
    --target-token-num ${TOKEN}

python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl
