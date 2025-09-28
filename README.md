
<div align="center">
  <h1>AutoPrune [NeurIPS 2025]</h1>
  <h3>
    <a href="" target="_blank" rel="noopener">
      ⚡️ AutoPrune: Each Complexity Deserves a Pruning Policy
    </a>
  </h3>
  <p>
    <p>
  <a href="https://veritas12.github.io/" target="_blank" rel="noopener">Hanshi Wang</a><sup>1,2,3,5*</sup>,
  <a href="https://openreview.net/profile?id=%7EYuhao_Xu5" target="_blank" rel="noopener">Yuhao Xu</a><sup>7</sup>,
  <a href="http://www.aisensors.ac.cn/author/%E5%BE%90%E6%B3%BD%E5%9D%A4/" target="_blank" rel="noopener">Zekun Xu</a><sup>1,2</sup>,
  <a href="https://nlpr.ia.ac.cn/users/gaojin/index.htm" target="_blank" rel="noopener">Jin Gao</a><sup>1,2,5†</sup>,
  <a href="https://people.ucas.ac.cn/~huweiming" target="_blank" rel="noopener">Weiming Hu</a><sup>1,2,5,6</sup>,
  <a href="https://zhipengzhang.cn/" target="_blank" rel="noopener">Zhipeng Zhang</a><sup>3,4†</sup>
</p>
  </p>
  <p>
    <sup>1</sup>State Key Laboratory of Multimodal Artificial Intelligence Systems (MAIS), CASIA<br/>
    <sup>2</sup>School of Artificial Intelligence, University of Chinese Academy of Sciences<br/>
    <sup>3</sup>School of Artificial Intelligence, Shanghai Jiao Tong University
    <sup>4</sup>Anyverse Intelligence<br/>
    <sup>5</sup>Beijing Key Laboratory of Super Intelligent Security of Multi-Modal Information<br/>
    <sup>6</sup>School of Information Science and Technology, ShanghaiTech University,
    <sup>7</sup>Sichuan University<br/>
    
  </p>
  <p>
    <sup>*</sup>This work was completed during Hanshi’s remote internship at SJTU.
    <sup>†</sup>Corresponding author.
  </p>
  <p>
    <a href="mailto:hanshi.wang.cv@outlook.com">hanshi.wang.cv@outlook.com</a>,
    <a href="mailto:2022141460058@stu.scu.edu.cn">2022141460058@stu.scu.edu.cn</a>,
    <a href="mailto:xuzekun2025@ia.ac.cn">xuzekun2025@ia.ac.cn</a>,
    <a href="mailto:jin.gao@nlpr.ia.ac.cn">jin.gao@nlpr.ia.ac.cn</a>,
    <a href="mailto:wmhu@nlpr.ia.ac.cn">wmhu@nlpr.ia.ac.cn</a>,
    <a href="mailto:zhipeng.zhang.cv@outlook.com">zhipeng.zhang.cv@outlook.com</a>
  </p>
</div>


## 🔥 News

[2025.9.18] AutoPrune is accepted by NeurIPS 2025.

## 👁️ Overview

The established redundancy in visual tokens within large vision–language models (LVLMs) allows for pruning to effectively reduce their substantial computational demands. Empirical evidence from previous works indicates that visual tokens in later decoder stages receive less attention than shallow layers. Then, previous methods typically employ heuristics layer-specific pruning strategies where, although the number of tokens removed may differ across decoder layers, the overall pruning schedule is fixed and applied uniformly to all input samples and tasks, failing to align token elimination with the model’s holistic reasoning trajectory. Cognitive science indicates that human visual processing often begins with broad exploration to accumulate evidence before narrowing focus as the target becomes distinct. Our experiments reveal an analogous pattern in LVLMs. This observation strongly suggests that neither a fixed pruning schedule nor a heuristics layer-wise strategy can optimally accommodate the diverse complexities inherent in different inputs. To overcome this limitation, we introduce Complexity-Adaptive Pruning (AutoPrune), which is a training-free, plug-and-play framework that tailors pruning policies to varying sample and task complexities. Specifically, AutoPrune quantifies the mutual information between visual and textual tokens, and then projects this signal to a budget-constrained logistic retention curve. Each such logistic curve, defined by its unique shape, is shown to effectively correspond with the specific complexity of different tasks, and can easily guarantee adherence to a pre-defined computational constraints. We evaluate AutoPrune not only on standard vision-language tasks but also on Vision-Language-Action (VLA) models for autonomous driving. Notably, when applied to LLaVA-1.5-7B, our method prunes 89% of visual tokens and reduces inference FLOPs by 76.8%, but still retaining 96.7% of the original accuracy averaged over all tasks. This corresponds to a 9.1% improvement over the recent work PDrop (CVPR'2025), demonstrating the effectivenes.


## ⚙️ Installation

### 🏝️ Environment

1. Clone this repository.
```bash
https://github.com/AutoLab-SAI-SJTU/AutoPrune.git
cd AutoPrune
```

2. Install necessary packages.
```bash
conda create -n AutoPrune python=3.10 -y
conda activate AutoPrune
pip install -e .
```

3. (Optional) Install FlashAttention for further inference acceleration.
```bash
pip install flash-attn --no-build-isolation
```

### 📦️ Model

Download corresponding [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) checkpoints from [Hugging Face](https://huggingface.co/liuhaotian) 🤗:

| Version | LLM | Checkpoint |
|----------|:----------:|:-----------:|
| LLaVA-1.5 | Vicuna-7B | [liuhaotian/llava-v1.5-7b](https://huggingface.co/liuhaotian/llava-v1.5-7b) |
| LLaVA-1.5 | Vicuna-13B | [liuhaotian/llava-v1.5-13b](https://huggingface.co/liuhaotian/llava-v1.5-13b) |
| LLaVA-1.6 (LLaVA-NeXT) | Vicuna-7B | [liuhaotian/llava-v1.6-vicuna-7b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-7b) |
| LLaVA-1.6 (LLaVA-NeXT) | Vicuna-13B | [liuhaotian/llava-v1.6-vicuna-13b](https://huggingface.co/liuhaotian/llava-v1.6-vicuna-13b) |

### 📊 Data

Download each dataset according to [EVAL.md](EVAL.md).

## 📋️ Evaluation

Using TextVQA as an example (scripts/v1_5/eval/textvqa.sh), inference is controlled by a few hyperparameters that shape the visual‑token retention curve:

- `--visual-token-num`: Initial number of visual tokens produced by the vision tower (LLaVA‑1.5: 576; LLaVA‑1.6: 2880). This is an upper bound; pruning will dynamically reduce it.
- `--target-token-num`: Target visual‑token budget. In the scripts, the first positional argument `TOKEN` is passed here. Smaller values prune more aggressively; larger values keep more tokens.
- `--x0`: Horizontal shift of the logistic retention curve. Increasing `x0` delays strong pruning to later layers (keeping more tokens early); decreasing it starts shrinking earlier.
- `--k0` and `--gamma`: Control the MI‑adaptive slope of the curve.
  - Internally we compute `dynamic_k = max(-gamma * MI + k0, 0)` and use it as the slope of the logistic curve.
  - Intuition: `k0` sets the base steepness (larger → sharper), while `gamma` controls sensitivity to sample complexity (mutual information; larger → more sensitive).

How to run (direct Python invocation is equivalent to the shell script):
```bash
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
```

```bash
CUDA_VISIBLE_DEVICES=2 bash scripts/v1_5/eval/textvqa.sh 64
```

- The trailing `64` is the `TOKEN` argument; the script forwards it to `--target-token-num` as your visual‑token budget (smaller → more pruning).
- v1_5 scripts fix `--visual-token-num 576`; v1_6 scripts fix `--visual-token-num 2880`.

Tuning tip: Optimal settings may vary by dataset/task. You can tuning `--x0 / --k0 / --gamma` per dataset for the best results. In our method we did not perform fine-grained hyperparameter tuning in order to demonstrate robustness, so with proper tuning it is likely to surpass the results reported in our paper.


## 🏆Main Results

## 🎗️ Citation

If you find AutoPrune useful for your research and applications, please cite using this BibTeX:
```bibtex
@article{wang2025autoprune,
  title={Each Complexity Deserves a Pruning Policy},
  author={Hanshi Wang, Yuhao Xu, Zekun Xu, Jin Gao, Yufan Liu, Weiming Hu, Ke Wang, Zhipeng Zhang},
  journal={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## 🎟️ License

This project is released under the [Apache 2.0 license](LICENSE).

## 🎉 Acknowledgement

AutoPrune uses code from a few open source repositories. Without the efforts of these folks (and their willingness to release their implementations), AutoPrune would not be possible. We thanks these authors for their efforts!
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [FastV](https://github.com/pkunlp-icler/FastV)
- [FasterVLM](https://github.com/Theia-4869/FasterVLM)
- [SparseVLM](https://github.com/Gumpest/SparseVLMs)
