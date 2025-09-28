# #!/bin/bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1


# CKPT="llava-v1.5-7b"
# METHOD="AdaPruner"
# TOKEN=${1}
# PARAM="n_${TOKEN}"

# python -W ignore -m llava.eval.model_vqa_loader \
#     --model-path ./models/llava-v1.5-7b \
#     --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
#     --image-folder ./playground/data/eval/textvqa/train_images \
#     --answers-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl \
#     --visual-token-num ${TOKEN} \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m llava.eval.eval_textvqa \
#     --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
#     --result-file ./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}.jsonl
#!/bin/bash

CKPT="llava-v1.5-7b"
METHOD="AdaPruner"
TOKEN=${1}
EXTRA_TAG=${2:-default}  # 如果未提供第二个参数，默认为 default
PARAM="n_${TOKEN}_${EXTRA_TAG}"

# 获取时间戳
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# 构建输出文件名
OUT_JSONL="./playground/data/eval/textvqa/answers/${CKPT}/${METHOD}/${PARAM}_${TIMESTAMP}.jsonl"
OUT_TXT="${OUT_JSONL%.jsonl}.txt"

# 确保输出目录存在
mkdir -p "$(dirname "$OUT_JSONL")"

# 第一步：运行推理
python -W ignore -m llava.eval.model_vqa_loader \
    --model-path ./models/llava-v1.5-7b \
    --question-file ./playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl \
    --image-folder ./playground/data/eval/textvqa/train_images \
    --answers-file "${OUT_JSONL}" \
    --visual-token-num 576 \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --x0 14.9 \
    --k0 0.4 \
    --gamma 0.2 \
    --target-token-num ${TOKEN}

# 第二步：运行评估，并把输出保存为 txt（同时打印到命令行）
python -m llava.eval.eval_textvqa \
    --annotation-file ./playground/data/eval/textvqa/TextVQA_0.5.1_val.json \
    --result-file "${OUT_JSONL}"
