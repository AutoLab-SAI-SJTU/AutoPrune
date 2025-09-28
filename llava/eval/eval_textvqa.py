import os
import argparse
import json
import re

from llava.eval.m4c_evaluator import TextVQAAccuracyEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str)
    parser.add_argument('--result-file', type=str)
    parser.add_argument('--result-dir', type=str)
    # 是否使用list形式
    parser.add_argument('--use_list', action='store_true', default=False)
    return parser.parse_args()


def prompt_processor(prompt):
    if prompt.startswith('OCR tokens: '):
        pattern = r"Question: (.*?) Short answer:"
        match = re.search(pattern, prompt, re.DOTALL)
        question = match.group(1)
    elif 'Reference OCR token: ' in prompt and len(prompt.split('\n')) == 3:
        if prompt.startswith('Reference OCR token:'):
            question = prompt.split('\n')[1]
        else:
            question = prompt.split('\n')[0]
    elif len(prompt.split('\n')) == 2:
        question = prompt.split('\n')[0]
    else:
        assert False

    return question.lower()


def eval_single(annotation_file, result_file):
    experiment_name = os.path.splitext(os.path.basename(result_file))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for result in results:
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        pred_list.append({
            "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))

def eval_list_single(annotation_file, result_file_list):
    experiment_name = os.path.splitext(os.path.basename(result_file_list[0]))[0]
    print(experiment_name)
    annotations = json.load(open(annotation_file))['data']
    annotations = {(annotation['image_id'], annotation['question'].lower()): annotation for annotation in annotations}
    result_list = []
    for i, result_file in enumerate(result_file_list):
        print(result_file)
        results = [json.loads(line) for line in open(result_file)]
        result_list.append(results)
    # results = [json.loads(line) for line in open(result_file)]

    pred_list = []
    for id_x, result in enumerate(result_list[0]):
        annotation = annotations[(result['question_id'], prompt_processor(result['prompt']))]
        result_text_list = []
        experiment_name_list = []
        for i, result_file in enumerate(result_file_list):
            results = result_list[i]
            result_text = results[id_x]['text']
            result_text_list.append(result_text)
            experiment_name = os.path.splitext(os.path.basename(result_file))[0]
            experiment_name_list.append(experiment_name)
        pred_list.append({
            "pred_answer": result_text_list,
            # "pred_answer": result['text'],
            "gt_answers": annotation['answers'],
            "experiment_name": experiment_name_list
        })

    evaluator = TextVQAAccuracyEvaluator()
    print('Samples: {}\nAccuracy: {:.2f}%\n'.format(len(pred_list), 100. * evaluator.eval_pred_list(pred_list)))


if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        if args.use_list:
            # eval_list_single(args.annotation_file, args.result_file)
            jsonl_dir = './playground/data/eval/textvqa/answers/all_K'
            pattern = re.compile(r'target(\d+)_A([\d.]+)_k([\d.eE+-]+)_x0([\d.]+)\.jsonl')
            file_name_list = os.listdir(jsonl_dir)
            use_target_token_num = 29
            use_file_name_with_k = []  # 用来保存 (文件名, k) 元组

            for filename in file_name_list:
                match = pattern.match(filename)
                if match:
                    target = int(match.group(1))
                    A = round(float(match.group(2)), 2)
                    k = float(match.group(3))  # 这里用 float 保持数值精度
                    x0 = round(float(match.group(4)), 2)

                    if target == use_target_token_num:
                        use_file_name_with_k.append((filename, k))
                        print(f"target_token_num: {target}, A: {A}, k: {k}, x0: {x0}")
            use_file_name_with_k.sort(key=lambda x: x[1])
            use_file_name_list_sorted = [item[0] for item in use_file_name_with_k]
            eval_list_single(args.annotation_file, [os.path.join(jsonl_dir, filename) for filename in use_file_name_list_sorted])
            # eval_list_single(args.annotation_file, [
            #     '/home/hswang/paper/llm/FasterVLM/playground/data/eval/textvqa/answers/llava-v1.5-7b/fastervlm/n_144_logistic_A75_K27_20250407_213910.jsonl',
            #     '/home/hswang/paper/llm/FasterVLM/playground/data/eval/textvqa/answers/llava-v1.5-7b/fastervlm/n_144_logistic_A75_K30_x15_merge1_top20_20250408_115954.jsonl',
            #     '/home/hswang/paper/llm/FasterVLM/playground/data/eval/textvqa/answers/llava-v1.5-7b/fastervlm/n_144_logistic_A75_K40_20250407_204705.jsonl'])
        else:
            eval_single(args.annotation_file, args.result_file)


    if args.result_dir is not None:
        for result_file in sorted(os.listdir(args.result_dir)):
            if not result_file.endswith('.jsonl'):
                print(f'Skipping {result_file}')
                continue
            eval_single(args.annotation_file, os.path.join(args.result_dir, result_file))
