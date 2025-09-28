# Evaluation

We evaluate AutoPrune with different [LLaVA](https://github.com/haotian-liu/LLaVA) models on a diverse set of 10 benchmarks. To ensure the reproducibility, we evaluate the models with greedy decoding following the originial LLaVA.

## Scripts

Before preparing task-specific data, **you MUST first download [eval.zip](https://drive.google.com/file/d/1atZSBBrAX54yYpxtVVW33zFvcnaHeFPy/view?usp=sharing)**. It contains custom annotations, scripts, and the prediction files with vanilla LLaVA-1.5. Extract it to `./playground/data/eval`. This also provides a general structure for all datasets.

### VQAv2

1. Download [`test2015`](http://images.cocodataset.org/zips/test2015.zip) and put it under `./playground/data/eval/vqav2`.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_5/eval/vqav2.sh 64
```
3. Submit the results to the [evaluation server](https://eval.ai/web/challenges/challenge-page/830/my-submission): `./playground/data/eval/vqav2/answers_upload`.

### GQA

1. Download the [data](https://cs.stanford.edu/people/dorarad/gqa/download.html) and [evaluation scripts](https://cs.stanford.edu/people/dorarad/gqa/evaluate.html) following the official instructions and put under `./playground/data/eval/gqa/data`. You may need to modify `eval.py` as [this](https://gist.github.com/haotian-liu/db6eddc2a984b4cbcc8a7f26fd523187) due to the missing assets in the GQA v1.2 release.
2. Multi-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3,4 bash scripts/v1_5/eval/gqa.sh 64
```


### ScienceQA

1. Under `./playground/data/eval/scienceqa`, download `images`, `pid_splits.json`, `problems.json` from the `data/scienceqa` folder of the ScienceQA [repo](https://github.com/lupantech/ScienceQA).
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/sqa.sh 64
```

### TextVQA

1. Download [`TextVQA_0.5.1_val.json`](https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json) and [images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip) and extract to `./playground/data/eval/textvqa`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/textvqa.sh 64
```

### POPE

1. Download `coco` from [POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco) and put under `./playground/data/eval/pope`.
2. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/pope.sh 64
```

### MME

1. Download the data following the official instructions [here](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation).
2. Downloaded images to `MME_Benchmark_release_version`.
3. put the official `eval_tool` and `MME_Benchmark_release_version` under `./playground/data/eval/MME`.
4. Single-GPU inference and evaluate.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mme.sh 64
```

### MMBench

1. Download [`mmbench_dev_20230712.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_20230712.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench.sh 64
```
3. Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_20230712`.

### MMBench-CN

1. Download [`mmbench_dev_cn_20231003.tsv`](https://download.openmmlab.com/mmclassification/datasets/mmbench/mmbench_dev_cn_20231003.tsv) and put under `./playground/data/eval/mmbench`.
2. Single-GPU inference.
```Shell
CUDA_VISIBLE_DEVICES=0 bash scripts/v1_5/eval/mmbench_cn.sh 64
```
3. Submit the results to the [evaluation server](https://mmbench.opencompass.org.cn/mmbench-submission): `./playground/data/eval/mmbench/answers_upload/mmbench_dev_cn_20231003`.


## Scripts with LLaVA-NeXT (LLaVA-1.6)

To evaluate AutoPrune with LLaVA-NeXT, you just need to replace the `v1_5` with `v1_6` in the shell scripts. For example, to evaluate VQAv2 with LLaVA-NeXT, you can run:

```Shell
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/v1_6/eval/vqav2.sh 160
```

## Results

### LLaVA-1.5-7B

<table>
  <thead>
    <tr>
      <th>Method</th><th>Present at</th><th>Avg. tokens</th><th>MME</th><th>MMB</th><th>SQA</th><th>GQA</th><th>TextVQA</th><th>Ratio</th><th>FLOPs</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>LLaVA-1.5-7B</b></td><td>NeurIPS’24</td><td>576</td><td>1862</td><td>64.7</td><td>69.5</td><td>61.9</td><td>58.2</td><td>100%</td><td>100%</td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="10"><b>Avg. tokens 192</b></td></tr>
    <tr><td>ToMe [15]</td><td>arXiv’22</td><td>192</td><td>1563</td><td>60.5</td><td>65.2</td><td>54.3</td><td>52.1</td><td>89.9%</td><td>44.3%</td></tr>
    <tr><td>FastV [18]</td><td>ECCV’24</td><td>192</td><td>1612</td><td>61.2</td><td>67.3</td><td>52.7</td><td>52.5</td><td>90.6%</td><td>45.7%</td></tr>
    <tr><td>SparseVLM [20]</td><td>arXiv’24</td><td>192</td><td>1721</td><td>62.5</td><td>69.1</td><td>57.6</td><td>66.3</td><td>95.5%</td><td>46.3%</td></tr>
    <tr><td>PDrop [13]</td><td>CVPR’25</td><td>192</td><td>1797</td><td>63.3</td><td>69.2</td><td>57.3</td><td>56.5</td><td>96.8%</td><td>43.9%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>192</td><td><b>1832</b></td><td><b>64.9</b></td><td><b>69.6</b></td><td><b>60.4</b></td><td><b>57.7</b></td><td><b>99.0%</b></td><td><b>42.9%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="10"><b>Avg. tokens 128</b></td></tr>
    <tr><td>ToMe [15]</td><td>arXiv’22</td><td>128</td><td>1343</td><td>53.3</td><td>59.6</td><td>52.4</td><td>49.1</td><td>81.1%</td><td>35.1%</td></tr>
    <tr><td>FastV [18]</td><td>ECCV’24</td><td>128</td><td>1490</td><td>56.1</td><td>60.2</td><td>49.6</td><td>50.6</td><td>83.9%</td><td>36.8%</td></tr>
    <tr><td>SparseVLM [20]</td><td>arXiv’24</td><td>128</td><td>1696</td><td>60.0</td><td>67.1</td><td>56.0</td><td>54.9</td><td>93.0%</td><td>37.3%</td></tr>
    <tr><td>PDrop [13]</td><td>CVPR’25</td><td>128</td><td>1761</td><td>61.6</td><td>68.4</td><td>57.1</td><td>56.6</td><td>95.6%</td><td>35.1%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>128</td><td><b>1785</b></td><td><b>64.3</b></td><td><b>69.7</b></td><td><b>59.9</b></td><td><b>57.4</b></td><td><b>98.1%</b></td><td><b>33.7%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="10"><b>Avg. tokens 64</b></td></tr>
    <tr><td>ToMe [15]</td><td>arXiv’22</td><td>64</td><td>1138</td><td>43.7</td><td>50.0</td><td>48.6</td><td>45.3</td><td>70.5%</td><td>25.7%</td></tr>
    <tr><td>FastV [18]</td><td>ECCV’24</td><td>64</td><td>1256</td><td>48.0</td><td>51.1</td><td>46.1</td><td>47.8</td><td>73.7%</td><td>27.9%</td></tr>
    <tr><td>SparseVLM [20]</td><td>arXiv’24</td><td>64</td><td>1505</td><td>56.2</td><td>62.2</td><td>52.7</td><td>51.8</td><td>85.9%</td><td>28.2%</td></tr>
    <tr><td>PDrop [13]</td><td>CVPR’25</td><td>64</td><td>1561</td><td>58.8</td><td>69.0</td><td>47.5</td><td>50.6</td><td>87.6%</td><td>25.5%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>64</td><td><b>1745</b></td><td><b>63.6</b></td><td><b>69.6</b></td><td><b>57.7</b></td><td><b>57.1</b></td><td><b>96.7%</b></td><td><b>23.2%</b></td></tr>
  </tbody>
</table>

<style>
  /* Basic look close to your example */
  table.prune { border-collapse: collapse; width: 100%; }
  .prune th, .prune td { border: 1px solid #444; padding: 6px 10px; text-align: center; }
  .prune thead th { background: #1f1f1f; color: #fff; }

  /* Group separator row like “Avg. tokens …” */
  .prune tr.group td { background: #2b2b2b; color: #fff; font-weight: 700; text-align: left; }

  /* Remove any background that a theme or zebra style might apply to the Ours rows */
  .prune tr.ours {
    background: transparent !important;
    background-color: transparent !important;
    background-image: none !important;
    color: inherit;
    font-weight: 700; /* keep the bold effect without a fill */
  }
  .left { text-align: left; }
</style>

<!-- Table 1: LLaVA-1.5-7B pruning (Avg. tokens 128, 64) -->
<table>
  <thead>
    <tr>
      <th>Method</th><th>Present at</th><th>Avg. tokens</th>
      <th>GQA</th><th>SQA</th><th>TextVQA</th><th>POPE</th><th>MME</th><th>MMB</th><th>MMB<sup>CN</sup></th><th>Ratio (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>LLaVA-1.5-7B</b></td><td>NeurIPS’24</td><td>576</td><td>61.9</td><td>69.5</td><td>58.2</td><td>85.9</td><td>1511</td><td>64.7</td><td>58.3</td><td>100.0</td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="11"><b>Avg. tokens 128</b></td></tr>
    <tr><td>ToMe</td><td>ICLR’23</td><td>128</td><td>52.4</td><td>59.6</td><td>49.1</td><td>62.8</td><td>1088</td><td>53.3</td><td>48.8</td><td>80.9%</td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>128</td><td>49.6</td><td>60.2</td><td>50.6</td><td>59.6</td><td>1209</td><td>56.1</td><td>51.4</td><td>82.6%</td></tr>
    <tr><td>SparseVLM</td><td>ICML’25</td><td>128</td><td>56.0</td><td>67.1</td><td>54.9</td><td>80.5</td><td>1376</td><td>60.0</td><td>51.1</td><td>92.4%</td></tr>
    <tr><td>PruMerge+</td><td>arXiv’24</td><td>128</td><td>57.8</td><td>67.6</td><td>54.3</td><td>81.5</td><td>1421</td><td>61.3</td><td>54.7</td><td>94.5%</td></tr>
    <tr><td>VisionZip</td><td>CVPR’25</td><td>128</td><td>57.6</td><td>68.9</td><td>56.8</td><td>83.2</td><td>1432</td><td>62.0</td><td>56.7</td><td>96.4%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>128</td><td>58.2</td><td>69.1</td><td>57.0</td><td>84.6</td><td><b>1461</b></td><td>62.7</td><td>57.3</td><td>97.4%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>128</td><td><b>59.9</b></td><td><b>69.6</b></td><td><b>57.7</b></td><td><b>85.2</b></td><td>1458</td><td><b>64.3</b></td><td><b>58.3</b></td><td><b>98.7%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="11"><b>Avg. tokens 64</b></td></tr>
    <tr><td>ToMe</td><td>ICLR’23</td><td>64</td><td>48.6</td><td>50.0</td><td>45.3</td><td>52.5</td><td>922</td><td>43.7</td><td>38.9</td><td>69.2%</td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>64</td><td>46.1</td><td>51.1</td><td>47.8</td><td>48.0</td><td>1020</td><td>48.0</td><td>42.7</td><td>71.6%</td></tr>
    <tr><td>SparseVLM</td><td>ICML’25</td><td>64</td><td>52.7</td><td>62.2</td><td>51.8</td><td>75.1</td><td>1221</td><td>56.2</td><td>46.1</td><td>85.4%</td></tr>
    <tr><td>PruMerge+</td><td>arXiv’24</td><td>64</td><td>54.9</td><td>68.6</td><td>53.0</td><td>77.4</td><td>1198</td><td>59.3</td><td>51.0</td><td>89.6%</td></tr>
    <tr><td>VisionZip</td><td>CVPR’25</td><td>64</td><td>55.1</td><td>69.0</td><td>55.5</td><td>77.0</td><td>1366</td><td>60.1</td><td>55.4</td><td>93.1%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>64</td><td>55.4</td><td>69.1</td><td>55.8</td><td>80.4</td><td>1370</td><td>61.3</td><td>55.1</td><td>94.0%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>64</td><td><b>57.7</b></td><td><b>69.6</b></td><td><b>57.1</b></td><td><b>82.5</b></td><td><b>1445</b></td><td><b>63.6</b></td><td><b>57.1</b></td><td><b>97.1%</b></td></tr>
  </tbody>
</table>




### LLaVA-1.5-13B

<table>
  <thead>
    <tr>
      <th>Method</th><th>Present at</th><th># Token</th>
      <th>VQA<sup>V2</sup></th><th>GQA</th><th>TextVQA</th><th>POPE</th><th>MME</th><th>Ratio (%)</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>LLaVA-1.5-13B</b></td><td>-</td><td>576</td><td>80.0</td><td>63.3</td><td>61.2</td><td>86.0</td><td>1531</td><td>100%</td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="9"><b>Tokens 288</b></td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>288</td><td>79.5</td><td>62.6</td><td><b>60.9</b></td><td>85.2</td><td><b>1545</b></td><td>99.6%</td></tr>
    <tr><td>SparseVLM</td><td>ICML’25</td><td>288</td><td>78.5</td><td>59.9</td><td>59.5</td><td>71.3</td><td>1497</td><td>94.1%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>288</td><td>79.0</td><td>61.0</td><td>60.0</td><td>86.0</td><td>1530</td><td>98.6%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>288</td><td><b>79.8</b></td><td><b>63.0</b></td><td><b>60.9</b></td><td><b>86.1</b></td><td>1530</td><td><b>99.8%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="9"><b>Tokens 144</b></td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>144</td><td>77.2</td><td>59.9</td><td>60.0</td><td>79.4</td><td>1494</td><td>95.8%</td></tr>
    <tr><td>SparseVLM</td><td>ICML’25</td><td>144</td><td>76.1</td><td>58.0</td><td>57.9</td><td>68.6</td><td>1499</td><td>91.8%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>144</td><td>77.4</td><td>58.7</td><td>59.0</td><td>83.1</td><td>1467</td><td>95.7%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>144</td><td><b>79.0</b></td><td><b>61.5</b></td><td><b>60.2</b></td><td><b>86.7</b></td><td><b>1506</b></td><td><b>98.7%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="9"><b>Tokens 58</b></td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>58</td><td>70.3</td><td>54.9</td><td>55.6</td><td>67.3</td><td>1360</td><td>86.5%</td></tr>
    <tr><td>SparseVLM</td><td>ICML’25</td><td>58</td><td>68.3</td><td>54.4</td><td>52.6</td><td>62.6</td><td>1285</td><td>82.8%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>58</td><td>73.1</td><td>56.0</td><td>57.4</td><td>74.7</td><td>1371</td><td>90.0%</td></tr>
    <tr><td><b>Ours</b></td><td>-</td><td>58</td><td><b>77.2</b></td><td><b>58.5</b></td><td><b>59.0</b></td><td><b>83.6</b></td><td><b>1478</b></td><td><b>95.8%</b></td></tr>
  </tbody>
</table>


### LLaVA-NeXT-7B

<table>
  <thead>
    <tr>
      <th>Method</th><th>Present at</th><th>Tokens</th><th>VQA<sup>V2</sup></th><th>GQA</th><th>TextVQA</th><th>POPE</th><th>MME</th><th>Ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr><td><b>LLAVA-NeXT-7B</b></td><td>NeurIPS’24</td><td>2880</td><td>81.2</td><td>62.9</td><td>59.6</td><td>86.3</td><td>1513.8</td><td>100.0%</td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="9"><b>Tokens 640</b></td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>640</td><td>78.9</td><td>60.4</td><td>58.4</td><td>83.1</td><td>1477.3</td><td>97.0%</td></tr>
    <tr><td>SparseVLM</td><td>arXiv’24</td><td>640</td><td>78.2</td><td>59.1</td><td>56.2</td><td>80.9</td><td>1456.3</td><td>94.9%</td></tr>
    <tr><td>VisionZip</td><td>CVPR’25</td><td>640</td><td>79.2</td><td>60.1</td><td>58.5</td><td>82.2</td><td>1468.4</td><td>96.7%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>640</td><td>79.8</td><td>61.6</td><td>59.3</td><td>85.9</td><td>1480.7</td><td>98.6%</td></tr>
    <tr><td><b>Ours</b></td><td>–</td><td>640</td><td><b>80.5</b></td><td><b>62.6</b></td><td><b>59.6</b></td><td><b>86.7</b></td><td><b>1515.7</b></td><td><b>99.7%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="9"><b>Tokens 320</b></td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>320</td><td>71.9</td><td>55.9</td><td>55.7</td><td>71.7</td><td>1282.9</td><td>87.7%</td></tr>
    <tr><td>SparseVLM</td><td>arXiv’24</td><td>320</td><td>71.4</td><td>56.5</td><td>52.4</td><td>73.5</td><td>1342.7</td><td>87.9%</td></tr>
    <tr><td>VisionZip</td><td>CVPR’25</td><td>320</td><td>74.2</td><td>58.1</td><td>55.3</td><td>75.0</td><td>1348.8</td><td>90.5%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>320</td><td>75.7</td><td>58.4</td><td>57.6</td><td>80.4</td><td>1370.1</td><td>93.3%</td></tr>
    <tr><td><b>Ours</b></td><td>–</td><td>320</td><td><b>78.9</b></td><td><b>61.3</b></td><td><b>59.5</b></td><td><b>85.6</b></td><td><b>1471.6</b></td><td><b>98.2%</b></td></tr>
    <tr style="background:#2b2b2b;color:#fff;"><td colspan="9"><b>Tokens 160</b></td></tr>
    <tr><td>FastV</td><td>ECCV’24</td><td>160</td><td>61.8</td><td>49.8</td><td>51.9</td><td>51.7</td><td>1079.5</td><td>74.7%</td></tr>
    <tr><td>SparseVLM</td><td>arXiv’24</td><td>160</td><td>62.2</td><td>50.2</td><td>45.1</td><td>54.6</td><td>1167.1</td><td>74.9%</td></tr>
    <tr><td>VisionZip</td><td>CVPR’25</td><td>160</td><td>67.3</td><td>54.3</td><td>54.7</td><td>59.4</td><td>1239.7</td><td>82.3%</td></tr>
    <tr><td>FasterVLM</td><td>arXiv’24</td><td>160</td><td>70.6</td><td>54.7</td><td>56.0</td><td>72.9</td><td>1226.0</td><td>86.7%</td></tr>
    <tr><td><b>Ours</b></td><td>–</td><td>160</td><td><b>76.4</b></td><td><b>59.4</b></td><td><b>57.2</b></td><td><b>81.4</b></td><td><b>1457.0</b></td><td><b>94.9%</b></td></tr>
  </tbody>
</table>
