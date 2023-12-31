# Copyright (c) OpenMMLab. and affiliates.

import argparse
import json
import os
from collections import defaultdict
import pandas as pd
from mixtralkit.mixtral import Mixtral


def parse_args():
    parser = argparse.ArgumentParser(description='Run an inference of mixtral-8x7b model')
    parser.add_argument('-m',
                        '--model-weights',
                        help='Model weights.',
                        default=None,
                        type=str)
    parser.add_argument('-t',
                        '--tokenizer',
                        help='path of tokenizer file.',
                        default=None,
                        type=str)
    parser.add_argument('--num-gpus', type=int)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    max_batch_size = 4
    max_seq_len = 1024
    max_gen_len = 1024

    generator = Mixtral.build(
        ckpt_dir=args.model_weights,
        tokenizer_path=args.tokenizer,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_gpus=args.num_gpus,
    )

    mmlu_path = "../mmlu"
    mmlu_files = os.listdir(mmlu_path)

    for file in mmlu_files:

        task = file.rstrip('.csv')
        file_path = mmlu_path + '/' + file
        df = pd.read_csv(file_path, header=None, usecols=[0])

        if os.path.exists("/workspace/MixtralKit/output_data.json"):
            os.remove("/workspace/MixtralKit/output_data.json")

        # 初始化一个字典，用于存储每一层的统计结果
        layer_stats = {layer: defaultdict(int) for layer in range(1, 33)}

        for prompts in df:
            '''
            prompts = [
                "Chaos isn't a pit, Chaos is a ladder.",
                ]
            '''
            temperature = 1.0 # for greedy decoding
            top_p = 0.9

            
            results = generator.text_completion(
                prompts,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )
            for prompt, result in zip(prompts, results):
                print("="*30 + "Example START" + "="*30 + '\n')
                print("[Prompt]:\n{}\n".format(prompt))
                print("[Response]:\n{}\n".format(result['generation']))
                print("="*30 + "Example END" + "="*30 + '\n')
            
            with open("/workspace/MixtralKit/output_data.json", "r") as file:
                for i, line in enumerate(file):
                    data = json.loads(line)
                    expert_indices = data['expert_indices']

                    for pair in expert_indices:
                        for number in pair:
                            layer_stats[(i % 32) + 1][number] += 1

            for layer in range(1, 33):
                print(f"Layer {layer}: {dict(layer_stats[layer])}")

            os.remove("/workspace/MixtralKit/output_data.json")

        layer_stats_json = {layer: dict(layer_stats[layer]) for layer in layer_stats}

        layer_stats_output_file = '/workspace/mmlu_result/' + task + '.json'
        with open(layer_stats_output_file,'a') as outfile:
            json.dump(layer_stats_json, outfile)
        
        print(f"Task {task} is  done")

if __name__ == "__main__":
    main()