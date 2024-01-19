# Copyright (c) OpenMMLab. and affiliates.

import argparse
import json
import os
import time
from collections import defaultdict
import pandas as pd
from mixtralkit.mixtral import Mixtral
from hqq.core.quantize import *
from hqq.models.hf.mixtral import MixtralHQQ


def parse_args():
    parser = argparse.ArgumentParser(description='Run an inference of mixtral-8x7b model')
    parser.add_argument('-M',
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

def init(args):
    max_batch_size = 1
    max_seq_len = 1024
    
    generator = Mixtral.build(
        ckpt_dir=args.model_weights,
        tokenizer_path=args.tokenizer,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        num_gpus=args.num_gpus,
    )

    return generator

def patch_linear_fct(linear_layer, quant_config):
	return HQQLinear(linear_layer, quant_config)

def quant(generator):
    #HQQ - 4bit - No stats compression
    
    patch_params   = {}
    attn_prams     = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=False)
    experts_params = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=False)
    
    #HQQ - 4bit - Compress stats (z_g256)
    '''
    patch_params = {}
    attn_prams     = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True)
    attn_prams['scale_quant_params']['group_size'] = 256
    experts_params = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True)
    experts_params['scale_quant_params']['group_size'] = 256
    '''

    #HQQ_4-bit_g64_z_s256 (attn)/ HQQ_2bit_g16_z_s128 (experts):   
    '''
    patch_params   = {}
    attn_prams     = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=True, quant_scale=True) 
    attn_prams['scale_quant_params']['group_size'] = 256
    experts_params = BaseQuantizeConfig(nbits=2, group_size=16, quant_zero=True, quant_scale=True) 
    '''

    #Attn
    patch_params['self_attn.q_proj'] = attn_prams
    patch_params['self_attn.k_proj'] = attn_prams
    patch_params['self_attn.v_proj'] = attn_prams
    patch_params['self_attn.o_proj'] = attn_prams
    #Experts
    patch_params['block_sparse_moe.experts.w1'] = experts_params
    patch_params['block_sparse_moe.experts.w2'] = experts_params
    patch_params['block_sparse_moe.experts.w3'] = experts_params

    MixtralHQQ.patch_model(generator, lambda l: l, patch_linear_fct, patch_params)

    return generator

def mmlu_eval(generator):

    max_gen_len = 1

    mmlu_path = "/workspace/mmlu"
    mmlu_files = os.listdir(mmlu_path)

    task_num = 0

    for csvfile in mmlu_files:

        task = csvfile.rstrip('.csv')
        file_path = mmlu_path + '/' + csvfile
        df = pd.read_csv(file_path, header=None, usecols=[0])

        if os.path.exists("/workspace/MixtralKit/output_data.json"):
            os.remove("/workspace/MixtralKit/output_data.json")

        # layer_predict_stats = {layer: defaultdict(int) for layer in range(1, 33)}
        # layer_hit_stats = {layer: defaultdict(int) for layer in range(1, 33)}
        # layer_actual_stats = {layer: defaultdict(int) for layer in range(1, 33)}

        prompt_num = 0

        print(f"Task {task} begins")

        for prompts in df[0]:
            '''
            prompts = [
                "Chaos isn't a pit, Chaos is a ladder.",
                ]
            '''
            prompts = [str(prompts)]
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
            
            if os.path.exists("/workspace/MixtralKit/output_data.json"):
                
                predict = [[] for _ in range(33)]
                actual = [[] for _ in range(33)]
                with open("/workspace/MixtralKit/output_data.json", "r") as file:
                    
                    for i, line in enumerate(file):
                        data = json.loads(line)
                        expert_indices = data['expert_indices']

                        j = i
                        if j % 63 != 62:
                            if (j%63) % 2 == 1: # actual
                                actual[((j % 63) + 1) >> 1] = expert_indices
                            else: # predict
                                predict[2 + ((j % 63) >> 1)] = expert_indices
                        elif j % 63 == 62: # actual layer 32
                            actual[32] = expert_indices

                seqlen = len(actual[0])
                # sentenceID, is_prompt=1, prompt_len=seq_len, token_ID(0 ~ seq_len-1), layerID(1 ~ 32), expert_list([])
                # sentenceID, is_prompt=1, prompt_len=seq_len, token_ID(0 ~ seq_len-1), layer_list([i, i+1], 1<=i<=31), expert_list([[],[]])
                for token_ID in range(seqlen):
                    for layer_ID in range(1,33):
                        output_str_actual  = str(task_num*64 + prompt_num) + ' ' + '1' + ' ' + str(seqlen) + ' ' + str(token_ID) + ' ' + str(layer_ID) + ' ' + str(actual[layer_ID][token_ID])
                        predict_next = [-1, -1]
                        if layer_ID == 32:
                            pass
                        else:
                            predict_next = predict[layer_ID+1][token_ID]
                        output_str_predict = str(task_num*64 + prompt_num) + ' ' + '1' + ' ' + str(seqlen) + ' ' + str(token_ID) + ' ' + str([layer_ID, layer_ID+1]) + ' ' + str([actual[layer_ID][token_ID], predict_next])
                        print(output_str_actual)
                        print(output_str_predict)
                        with open("/workspace/MixtralKit/output_str_actual.txt", "a") as file:
                            file.write(output_str_actual)
                            file.write("\n")
                        with open("/workspace/MixtralKit/output_str_predict.txt", "a") as file:
                            file.write(output_str_predict)
                            file.write("\n")

                os.remove("/workspace/MixtralKit/output_data.json")

            prompt_num = prompt_num + 1
            if prompt_num == 64:
                break
        '''
        layer_predict_stats_json = {layer: dict(layer_predict_stats[layer]) for layer in layer_predict_stats}
        layer_hit_stats_json = {layer: dict(layer_hit_stats[layer]) for layer in layer_hit_stats}
        layer_actual_stats_json = {layer: dict(layer_actual_stats[layer]) for layer in layer_actual_stats}

        layer_stats_output_file = '/workspace/mmlu_result_predict/' + task + '.json'
        with open(layer_stats_output_file,'a') as outfile:
            json.dump(layer_predict_stats_json, outfile)
            outfile.write("\n")
            json.dump(layer_hit_stats_json, outfile)
            outfile.write("\n")
            json.dump(layer_actual_stats_json, outfile)
            outfile.write("\n")
        '''
        print(f"Task {task} is done")

        task_num = task_num + 1
        if task_num == 16:
            break

def main(generator):

    max_gen_len = 128

    prompts = [
        "Chaos isn't a pit, Chaos is a ladder. Are you going up or down?",
        ]
    
    temperature = 1.0 # for greedy decoding
    top_p = 0.9

    start_time = time.time()

    results = generator.text_completion(
        prompts,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
    )

    end_time = time.time()
    elapsed_time = end_time - start_time
    formatted_time = "{:.3f}".format(elapsed_time)
    print(f"generation time: {formatted_time} s")

    for prompt, result in zip(prompts, results):
        print("="*30 + "Example START" + "="*30 + '\n')
        print("[Prompt]:\n{}\n".format(prompt))
        print("[Response]:\n{}\n".format(result['generation']))
        print("="*30 + "Example END" + "="*30 + '\n')


if __name__ == "__main__":
    args = parse_args()
    generator = init(args)
    generator = quant(generator)
    # main(generator)
    mmlu_eval(generator)