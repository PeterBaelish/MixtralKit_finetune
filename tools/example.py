# Copyright (c) OpenMMLab. and affiliates.

import argparse
import json
import os
import time
import math
from collections import defaultdict
import pandas as pd
from pathlib import Path
import gzip
import requests
from tqdm import tqdm
import subprocess

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
    max_seq_len = 16384
    
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
    attn_prams     = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False)
    experts_params = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False)
    sparse_params  = BaseQuantizeConfig(nbits=4, group_size=64, quant_zero=False, quant_scale=False)
    
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
    #Sparse
    patch_params['sparse'] = sparse_params

    MixtralHQQ.patch_model(generator, lambda l: l, patch_linear_fct, patch_params)

    return generator

def mmlu_predict_test(generator):

    max_gen_len = 128

    mmlu_path = "/workspace/mmlu"
    mmlu_files = os.listdir(mmlu_path)

    task_num = 0

    for csvfile in mmlu_files:

        task = csvfile.rstrip('.csv')
        file_path = mmlu_path + '/' + csvfile
        df = pd.read_csv(file_path, header=None, usecols=[0])

        if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):
            os.remove("/workspace/MixtralKit_finetune/output_data.json")

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
            
            if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):
                
                predict = [[] for _ in range(33)]
                actual = [[] for _ in range(33)]
                
                with open("/workspace/MixtralKit_finetune/output_data.json", "r") as file:

                    prompt_len = 0
                    gen_len = 0
                    is_prompt = 1
                    
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

                        if j % 63 == 62:
                            
                            seqlen = len(actual[1])
                            if seqlen == 1:
                                is_prompt = 0
                                gen_len = gen_len + 1
                            else:
                                prompt_len = seqlen
                                
                            # sentenceID, is_prompt=0 or 1, prompt_len, token_ID(0 ~ seq_len-1), layerID(1 ~ 32), expert_list([])
                            # sentenceID, is_prompt=0 or 1, prompt_len, token_ID(0 ~ seq_len-1), layer_list([i, i+1], 1<=i<=31), expert_list([[],[]])
                            for token_ID in range(seqlen):
                                for layer_ID in range(1,33):
                                    
                                    token_pos = token_ID
                                    if is_prompt == 0:
                                        token_pos = prompt_len + gen_len - 1
                                    
                                    output_str_actual  = str(task_num*64 + prompt_num) + ' ' + str(is_prompt) + ' ' + str(prompt_len) + ' ' + str(token_pos) + ' ' + str(layer_ID) + ' ' + str(actual[layer_ID][token_ID])
                                    predict_next = [-1, -1]
                                    if layer_ID == 32:
                                        pass
                                    else:
                                        predict_next = predict[layer_ID+1][token_ID]
                                    output_str_predict = str(task_num*64 + prompt_num) + ' ' + str(is_prompt) + ' ' + str(prompt_len) + ' ' + str(token_pos) + ' ' + str([layer_ID, layer_ID+1]) + ' ' + str([actual[layer_ID][token_ID], predict_next])
                                    print(output_str_actual)
                                    print(output_str_predict)
                                    with open("/workspace/MixtralKit_finetune/output_str_actual.txt", "a") as file:
                                        file.write(output_str_actual)
                                        file.write("\n")
                                    with open("/workspace/MixtralKit_finetune/output_str_predict.txt", "a") as file:
                                        file.write(output_str_predict)
                                        file.write("\n")
                            predict = [[] for _ in range(33)]
                            actual = [[] for _ in range(33)]
                    
                os.remove("/workspace/MixtralKit_finetune/output_data.json")

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
        if task_num == 4:
            break

def mmlu_perplexity_test(generator):
    
    max_gen_len = 128

    mmlu_path = "/workspace/mmlu"
    mmlu_files = os.listdir(mmlu_path)

    token_num = 0
    log_sum = 0

    task_num = 0

    for csvfile in mmlu_files:

        task = csvfile.rstrip('.csv')
        file_path = mmlu_path + '/' + csvfile
        df = pd.read_csv(file_path, header=None, usecols=[0])

        if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):
            os.remove("/workspace/MixtralKit_finetune/output_data.json")

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
            
            if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):

                with open("/workspace/MixtralKit_finetune/output_data.json", "r") as file:

                    for i, line in enumerate(file):
                        data = json.loads(line) # List: [bsz]
                        for j in range(len(data)):
                            log_sum = log_sum + math.log(data[j])
                            token_num = token_num + 1

                os.remove("/workspace/MixtralKit_finetune/output_data.json")

            print("middle Perplexity = ", math.exp(-log_sum/token_num))

            prompt_num = prompt_num + 1
            if prompt_num == 64:
                break

        print(f"Task {task} is done")

        task_num = task_num + 1
        if task_num == 4:
            break
    
    print("Perplexity = ", math.exp(-log_sum/token_num))

def mmlu_performance_test(generator):

    PROMPT_PATH = '/workspace/mmlu/test_prompt.json'
    ANSWER_PATH = '/workspace/mmlu/test_standard_answer.json'
    task_list = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    prompts = {}
    answers = {}
    with open(PROMPT_PATH, 'r') as file:
        prompts = json.load(file)
    
    with open(ANSWER_PATH, 'r') as file:
        answers = json.load(file)

    max_gen_len = 1
    temperature = 1.0
    top_p = 0.9

    right_prompt = 0
    total_prompt = 0

    for task in task_list:

        prompt = prompts[task]

        task_right_prompt = 0
        task_total_prompt = 0

        for i, question in enumerate(prompt):

            question = [prompt[i]]
            answer = answers[task][i]

            result = generator.text_completion(
                question,
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
            )

            if result != []:

                print("="*30 + "Example START" + "="*30 + '\n')
                print("[Prompt]:\n{}\n".format(question))
                print("[Response]:\n{}\n".format(result[0]['generation']))

                if result[0]['generation'] == answer:
                    print("Answer Right")
                    right_prompt = right_prompt + 1
                    task_right_prompt = task_right_prompt + 1
                else:
                    print("Answer Wrong, Right is: ", answer)
                
                total_prompt = total_prompt + 1
                task_total_prompt = task_total_prompt + 1

                print("="*30 + "Example END" + "="*30 + '\n')
        
            print("Total prompt: ", total_prompt)
            print("Right prompt: ", right_prompt)
            print("Current Score: ", (right_prompt/total_prompt)*100)
            print("Current Task Score: ", (task_right_prompt/task_total_prompt)*100)
        
        print(f"Task {task} end.")
        print("Task score:", (task_right_prompt/task_total_prompt)*100)
    
    print("test end.")
    print("Total prompt: ", total_prompt)
    print("Right prompt: ", right_prompt)
    print("Score: ", (right_prompt/total_prompt)*100)

def winogrande_performance_test(generator):

    PROMPT_PATH = '/workspace/winogrande/winogrande.jsonl'

    prompts = []

    with open(PROMPT_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            dict_line = json.loads(line)
            prompts.append(dict_line)

    max_gen_len = 1
    temperature = 1.0
    top_p = 0.9

    right_prompt = 0
    total_prompt = 0

    for i in range(len(prompts)):

        prompt = prompts[i]

        question = "Please choose the most appropriate word to fill in the blank : " + prompt["sentence"] + " A) " + prompt["option1"] + " B) " + prompt["option2"] + ". Please choose one option and provide only A or B as your answer."
        question = [question]
        
        answer = "A" if prompt["answer"] == "1" else "B"

        result = generator.text_completion(
            question,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        if result != []:

            print("="*30 + "Example START" + "="*30 + '\n')
            print("[Prompt]:\n{}\n".format(question))
            print("[Response]:\n{}\n".format(result[0]['generation']))

            if result[0]['generation'] == answer:
                print("Answer Right, Right is: ", answer)
                right_prompt = right_prompt + 1
            else:
                print("Answer Wrong, Right is: ", answer)
            
            total_prompt = total_prompt + 1

            print("="*30 + "Example END" + "="*30 + '\n')
    
        print("Total prompt: ", total_prompt)
        print("Right prompt: ", right_prompt)
        print("Current Score: ", (right_prompt/total_prompt)*100)
    
    print("test end.")
    print("Total prompt: ", total_prompt)
    print("Right prompt: ", right_prompt)
    print("Score: ", (right_prompt/total_prompt)*100)

def piqa_performance_test(generator):

    PROMPT_PATH = '/workspace/piqa/dev.jsonl'
    ANSWER_PATH = '/workspace/piqa/dev-labels.lst'

    prompts = []
    answers = []

    with open(PROMPT_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            dict_line = json.loads(line)
            prompts.append(dict_line)
    
    with open(ANSWER_PATH, 'r', encoding='utf-8') as file:
        for line in file:
            number = int(line.strip())  # strip()去除可能的空白符，如换行符
            answers.append(number)

    max_gen_len = 1
    temperature = 1.0
    top_p = 0.9

    right_prompt = 0
    total_prompt = 0

    for i in range(len(prompts)):

        prompt = prompts[i]
        answer = answers[i]

        question = "Given the goal: " + '"' + prompt["goal"] + '"' +  ", which of the following solutions is the best? " + " A) " + prompt["sol1"] + " B) " + prompt["sol2"] + ". Please choose one solution and provide only A or B as your answer."
        question = [question]

        answer = "A" if answer == 0 else "B"

        ''''''
        result = generator.text_completion(
            question,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        if result != []:

            print("="*30 + "Example START" + "="*30 + '\n')
            print("[Prompt]:\n{}\n".format(question))
            print("[Response]:\n{}\n".format(result[0]['generation']))

            if result[0]['generation'] == answer:
                print("Answer Right, Right is: ", answer)
                right_prompt = right_prompt + 1
            else:
                print("Answer Wrong, Right is: ", answer)
            
            total_prompt = total_prompt + 1

            print("="*30 + "Example END" + "="*30 + '\n')
    
        print("Total prompt: ", total_prompt)
        print("Right prompt: ", right_prompt)
        print("Current Score: ", (right_prompt/total_prompt)*100)
    
    print("test end.")
    print("Total prompt: ", total_prompt)
    print("Right prompt: ", right_prompt)
    print("Score: ", (right_prompt/total_prompt)*100)

def sparsity_wikipedia_dataset_output(generator):
    
    max_gen_len = 128

    # wikipedia: /workspace/wikipedia/20231101.*/train-00000-of-00001.parquet 
    '''
    #  {'id': '1',
        'url': 'https://simple.wikipedia.org/wiki/April',
        'title': 'April',
        'text': 'April is the fourth month...'
        }
    '''

    prompt_path = Path("/workspace/wikipedia")

    for p in prompt_path.rglob('*'):

        if p.is_dir():

            for prompt_file in p.rglob('*'):

                df = pd.read_parquet(prompt_file)

                print(df.head())

                if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):
                    os.remove("/workspace/MixtralKit_finetune/output_data.json")

                print(f"Task {prompt_file} begins")

                for index, row in df.iterrows():
                    
                    prompts = row['dict_cloumn']['text']
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
                    
                    if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):

                        with open("/workspace/MixtralKit_finetune/output_data.json", "r") as file:
                            data = []
                            for line in file:
                                data.append(json.loads(line))
                            for dict in data:
                                for i, expert_id in enumerate(dict["expert_id"]):
                                    dataset_path = "/workspace/Sparsity_Dataset/wikipedia/Layer" + str(dict["layer_id"]) + "_Expert" + str(expert_id) + ".json"
                                    dataset_dict = {
                                        "input_tensor": dict["input_tensor"],
                                        "sparsity": dict[f"expert{i}_sparsity"]
                                    }
                                    with open(dataset_path, "a") as dataset_file:
                                        dataset_string = json.dumps(dataset_dict)
                                        dataset_file.write(dataset_string)
                                        dataset_file.write("\n")

                        os.remove("/workspace/MixtralKit_finetune/output_data.json")

                print(f"Task {prompt_file} is done")

def sparsity_c4_dataset_output(generator):
    
    max_gen_len = 128

    # c4: /workspace/c4/en/c4-train.*-of-01024.json.gz
    '''
    {
    'url': 'https://klyq.com/beginners-bbq-class-taking-place-in-missoula/',
    'text': 'Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ? You will have the opportunity, put this on your calendar now. Thursday, September 22nd join World Class BBQ Champion, Tony Balay from Lonestar Smoke Rangers. He will be teaching a beginner level class for everyone who wants to get better with their culinary skills.\nHe will teach you everything you need to know to compete in a KCBS BBQ competition, including techniques, recipes, timelines, meat selection and trimming, plus smoker and fire information.\nThe cost to be in the class is $35 per person, and for spectators it is free. Included in the cost will be either a t-shirt or apron and you will be tasting samples of each meat that is prepared.',
    'timestamp': '2019-04-25T12:57:54Z'
    }
    '''

    prompt_path = Path("/workspace/c4/en")

    for prompt_file in sorted(prompt_path.rglob('*'), key=lambda x: x.name, reverse=True):

        df = []
        with gzip.open(prompt_file, "rt") as gz_file:
            for line in gz_file:
                df.append(json.loads(line))

        print(f"Task {prompt_file} begins")

        for prompt_dict in df:
            
            if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):
                os.remove("/workspace/MixtralKit_finetune/output_data.json")

            print(prompt_dict)

            prompts = prompt_dict['text']
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
            
            if os.path.exists("/workspace/MixtralKit_finetune/output_data.json"):

                with open("/workspace/MixtralKit_finetune/output_data.json", "r") as file:
                    data = []
                    for line in file:
                        data.append(json.loads(line))
                    for dict in data:
                        # print(dict)
                        for i, expert_id in enumerate(dict["expert_id"]):
                            dataset_path = "/workspace/Sparsity_Dataset/c4/Layer" + str(dict["layer_id"]) + "_Expert" + str(expert_id) + ".json"
                            dataset_dict = {
                                "input_tensor": dict["input_tensor"], # tensor (4096)
                                "sparsity": dict[f"expert{i}_sparsity"] # tensor (14336)
                            }
                            # print(dataset_dict)
                            with open(dataset_path, "a") as dataset_file:
                                dataset_string = json.dumps(dataset_dict)
                                dataset_file.write(dataset_string)
                                dataset_file.write("\n")

                os.remove("/workspace/MixtralKit_finetune/output_data.json")

        print(f"Task {prompt_file} is done")

def main(generator):

    max_gen_len = 128+1

    prompts = [
"An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is a real number that represents the confidence of the output being the correct answer to the query. The dot product attention function is a simple attention is a"
,]

# input 64 token
# text = "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is a real number that represents the confidence of the output being the correct answer to the query. The dot product attention function is a simple attention is a"
# input 128 token
# text = "LLM typically stands for 'Large Language Model'. It refers to a type of artificial intelligence model designed to understand and generate human language. These models are trained on vast amounts of text data and use complex neural network architectures to perform tasks such as language translation, text summarization, question answering, and more. Large Language Models, like GPT-4, are capable of generating coherent and contextually relevant responses based on the input they receive. They are utilized in various applications, including chatbots, virtual assistants, and automated content creation. The key features of LLMs include their ability to understand context, generate natural language text, and perform various language-related tasks"

    
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
    
    if os.path.exists("/workspace/MixtralKit_finetune/output_data.txt"):

        integers = []
        with open("/workspace/MixtralKit_finetune/output_data.txt", "r") as file:
            for line in file:
                integers.append(float(line.strip()))
        
        cache_predict_hit = [0]*32
        n_token = len(integers)//32

        for i in range(n_token):
            for j in range(32):
                cache_predict_hit[j] += integers[i*32+j]
        
        for i in range(32):
            cache_predict_hit[i] /= n_token
            cache_predict_hit[i] /= 2
        
        print(cache_predict_hit)

        os.remove("/workspace/MixtralKit_finetune/output_data.txt")

if __name__ == "__main__":
    args = parse_args()
    generator = init(args)
    generator = quant(generator)
    
    main(generator)

    result = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
        encoding='utf-8'
    )

    memory_info = result.strip().split('\n')
    info = memory_info[7]
    total, used, free = info.split(',')
    print(f"总内存: {total}MB, 已用内存: {used}MB, 空闲内存: {free}MB")

    # mmlu_perplexity_test(generator)
    # mmlu_predict_test(generator)
    # mmlu_performance_test(generator)
    # sparsity_wikipedia_dataset_output(generator)
    # sparsity_c4_dataset_output(generator)

    # winogrande_performance_test(generator)
    # piqa_performance_test(generator)
