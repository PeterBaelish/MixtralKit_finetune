# Copyright (c) OpenMMLab. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict
import safetensors
from safetensors.torch import load_file
from transformers import LlamaTokenizer
import subprocess

import torch
import torch.nn.functional as F

from mixtralkit.layers import (
    Tokenizer,
    MoETorchTransformer,
    MixtralModelArgs,
    PreloadMoETorchTransformer, # exp
    QuantMoETorchTransformer, 
    SingleGPUMoETorchTransformer, # exp
    SparsePredictMoETorchTransformer, 
    PruneSingleGPUMoETorchTransformer,
    JustSparseSingleGPUMoETorchTransformer,
    NueronCacheSingleGPUMoETorchTransformer,
    NeuronCachePreloadMoETorchTransformer, #exp
    MixQuantSingleGPUMoETorchTransformer,
    MixQuantNeuronCachePreloadMoETorchTransformer # 1500ms #exp
)
from mixtralkit.utils import sample_top_p
from mixtralkit.utils.generation import (
    CompletionPrediction,
    Dialog,
    ChatPrediction

)

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Mixtral:
    '''
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        num_gpus: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Mixtral":

        model_parallel_size = 1

        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()

        # load model in .safetensors
        checkpoints = sorted(Path(ckpt_dir).glob("*.safetensors"))
        ckpt_paths = checkpoints

        print(ckpt_paths)
        
        with open(Path(ckpt_dir) / "config.json", "r") as f:
            params = json.loads(f.read())

        model_args: MixtralModelArgs = MixtralModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            num_gpus=num_gpus,
            **params,
        )

        # tokenizer = Tokenizer(model_path=tokenizer_path)
        # model_args.vocab_size = tokenizer.n_words
        # print("tokenizer.n_words: ", tokenizer.n_words)
        tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, local_files_only=True) # THIS CAN HARM GPU!!
        model_args.vocab_size = 32002
        
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = MixQuantSingleGPUMoETorchTransformer(model_args)
        print(f"=== created Mixtral 8x7B. Experts spread over {num_gpus} GPUs ===")
        model_param_keys = []
        for key, value in model.named_parameters():
            model_param_keys.append(key)

        # checkpoint = torch.load(ckpt_path, map_location="cpu")
        checkpoint = {}
        for ckpt_path in ckpt_paths:
            checkpoint.update(load_file(ckpt_path)) # checkpoint is model in dict, ckpt_path is path to model

        print("Total number of model parameters:{}".format(len(model_param_keys)))
        print("Total number of checkpoint parameters:{}".format(len(checkpoint)))

        checkpoint = {k.partition('model.')[2] if k.startswith("model.") else k : v for k, v in checkpoint.items() }

        for name in checkpoint:
            print(name)
        
        # transpose w1,3 in FFN
        for i in range(32):
            for j in range(8):
                w1_weight = "layers." + str(i) + ".block_sparse_moe.experts." + str(j) + ".w1.weight"
                if w1_weight in checkpoint:
                    # print("transpose ", w1_weight)
                    weight = checkpoint[w1_weight]
                    t_weight = weight.t()
                    checkpoint[w1_weight] = t_weight
        
        for i in range(32):
            for j in range(8):
                w3_weight = "layers." + str(i) + ".block_sparse_moe.experts." + str(j) + ".w3.weight"
                if w3_weight in checkpoint:
                    # print("transpose ", w3_weight)
                    weight = checkpoint[w3_weight]
                    t_weight = weight.t()
                    checkpoint[w3_weight] = t_weight

        model.load_state_dict(checkpoint, strict=False)
        
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Mixtral(model, tokenizer)
    
    '''
    @staticmethod
    def build(
        ckpt_dir: str,
        tokenizer_path: str,
        max_seq_len: int,
        max_batch_size: int,
        num_gpus: int,
        model_parallel_size: Optional[int] = None,
        seed: int = 1,
    ) -> "Mixtral":

        model_parallel_size = 1

        # seed must be the same in all processes
        torch.manual_seed(seed)

        start_time = time.time()
        ckpt_paths = sorted(Path(ckpt_dir).glob("*.pth"))

        assert len(ckpt_paths) > 0, f"no checkpoint files found in {ckpt_dir}"

        with open(Path(ckpt_dir) / "params.json", "r") as f:
            params = json.loads(f.read())

        model_args: MixtralModelArgs = MixtralModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            num_gpus=num_gpus,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        model_args.vocab_size = tokenizer.n_words
        torch.set_default_tensor_type(torch.cuda.HalfTensor)
        model = NeuronCachePreloadMoETorchTransformer(model_args)
        print(f"=== created Mixtral 8x7B. Experts spread over {num_gpus} GPUs ===")
        model_param_keys = []
        for key, value in model.named_parameters():
            model_param_keys.append(key)

        print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        print(ckpt_paths)
        checkpoint = {}
        for ckpt_path in ckpt_paths:
            sp_weight = torch.load(ckpt_path, map_location="cpu")
            checkpoint.update(sp_weight)
        print("BBBBBBBBBBBBBBBBBBBBBBBBBBBB")

        for name in checkpoint:
            print(name)
        
        checkpoint = {k.partition('model.')[2] if k.startswith("model.") else k : v for k, v in checkpoint.items() }

        print("Total number of model parameters:{}".format(len(model_param_keys)))
        print("Total number of checkpoint parameters:{}".format(len(checkpoint)))

        # modify dict name
        checkpoint["embed_tokens.weight"] = checkpoint.pop("tok_embeddings.weight")
        checkpoint["norm.weight"] = checkpoint.pop("norm.weight")
        checkpoint["lm_head.weight"] = checkpoint.pop("output.weight")
        for i in range(32):
            checkpoint[f"layers.{i}.input_layernorm.weight"] = checkpoint.pop(f"layers.{i}.attention_norm.weight")
            checkpoint[f"layers.{i}.self_attn.q_proj.weight"] = checkpoint.pop(f"layers.{i}.attention.wq.weight")
            checkpoint[f"layers.{i}.self_attn.k_proj.weight"] = checkpoint.pop(f"layers.{i}.attention.wk.weight")
            checkpoint[f"layers.{i}.self_attn.v_proj.weight"] = checkpoint.pop(f"layers.{i}.attention.wv.weight")
            checkpoint[f"layers.{i}.self_attn.o_proj.weight"] = checkpoint.pop(f"layers.{i}.attention.wo.weight")
            checkpoint[f"layers.{i}.post_attention_layernorm.weight"] = checkpoint.pop(f"layers.{i}.ffn_norm.weight")
            checkpoint[f"layers.{i}.block_sparse_moe.gate.weight"] = checkpoint.pop(f"layers.{i}.feed_forward.gate.weight")
            for j in range(8):
                checkpoint[f"layers.{i}.block_sparse_moe.experts.{j}.w1.weight"] = checkpoint.pop(f"layers.{i}.feed_forward.experts.{j}.w1.weight")
                checkpoint[f"layers.{i}.block_sparse_moe.experts.{j}.w2.weight"] = checkpoint.pop(f"layers.{i}.feed_forward.experts.{j}.w2.weight")
                checkpoint[f"layers.{i}.block_sparse_moe.experts.{j}.w3.weight"] = checkpoint.pop(f"layers.{i}.feed_forward.experts.{j}.w3.weight")
        
        # transpose w1,3 in FFN
        for i in range(32):
            for j in range(8):
                w1_weight = "layers." + str(i) + ".block_sparse_moe.experts." + str(j) + ".w1.weight"
                if w1_weight in checkpoint:
                    print("transpose ", w1_weight)
                    weight = checkpoint[w1_weight]
                    t_weight = weight.t()
                    checkpoint[w1_weight] = t_weight
        
        for i in range(32):
            for j in range(8):
                w3_weight = "layers." + str(i) + ".block_sparse_moe.experts." + str(j) + ".w3.weight"
                if w3_weight in checkpoint:
                    print("transpose ", w3_weight)
                    weight = checkpoint[w3_weight]
                    t_weight = weight.t()
                    checkpoint[w3_weight] = t_weight

        model.load_state_dict(checkpoint, strict=False)
        print(f"Loaded in {time.time() - start_time:.2f} seconds")

        return Mixtral(model, tokenizer)
    

    def __init__(self, model: NeuronCachePreloadMoETorchTransformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    # @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        logprobs: bool = False,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        # assert max_prompt_len <= params.max_seq_len
        if max_prompt_len >= params.max_seq_len:
            return ([], None)

        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        # pad_id = self.tokenizer.pad_token_id
        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device="cuda") # tokens: cpu->gpu
        for k, t in enumerate(prompt_tokens):
            adj_len = min(len(t), total_len-1)
            min_prompt_len = min(min_prompt_len, adj_len)
            t = t[: adj_len]
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device="cuda")
        if logprobs:
            token_logprobs = torch.zeros_like(tokens, dtype=torch.float)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz, device="cuda")
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:
            logits = self.model.forward(tokens, prev_pos)
            token_logprobs = -F.cross_entropy(
                input=logits.transpose(1, 2),
                target=tokens,
                reduction="none",
                ignore_index=pad_id,
            )

        total_time = 0
        gen_len_set = 9

        for cur_pos in range(min_prompt_len, total_len):
            '''
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,nounits,noheader'],
                encoding='utf-8'
            )

            memory_info = result.strip().split('\n')
            info = memory_info[6]
            total, used, free = info.split(',')
            print(f"总内存: {total}MB, 已用内存: {used}MB, 空闲内存: {free}MB")
            '''
            #if cur_pos - min_prompt_len == gen_len_set:
            #    break
            '''
            print("==============================================================================")
            print("current_position:", cur_pos)
            print("current token:", self.tokenizer.decode(tokens[0, prev_pos].tolist()))
            '''
            start_time = time.time()
            
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)

            end_time = time.time()
            elapsed_time = (end_time - start_time) * 1000
            if prev_pos != 0:
                total_time += elapsed_time
            
            if prev_pos == 0:
                print(f"prefill time: {elapsed_time} ms")
            else:
                print(f"decode time(generate 1 token): {elapsed_time} ms")
            ''''''    
            '''
            probs, _ = torch.max(torch.softmax(logits[:, -1], dim=-1),dim=1) # probs: (bsz)

            print(probs.tolist())

            with open("/workspace/MixtralKit/output_data.json", "a") as file:
                json.dump(probs.tolist(), file)
                file.write("\n")
            '''
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            if logprobs:
                token_logprobs[:, prev_pos + 1 : cur_pos + 1] = -F.cross_entropy(
                    input=logits.transpose(1, 2),
                    target=tokens[:, prev_pos + 1 : cur_pos + 1],
                    reduction="none",
                    ignore_index=pad_id,
                )
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                # next_token == self.tokenizer.eos_token_id
                next_token == self.tokenizer.eos_id
            )

            prev_pos = cur_pos
            if all(eos_reached):
                break
            
            if prev_pos - min_prompt_len == 8 or prev_pos - min_prompt_len == 64:
                print(f"total time: {total_time} ms")
                print(f"gen len: {prev_pos - min_prompt_len}")
                print(f"average time: {total_time/(prev_pos - min_prompt_len)}ms")

        print(f"total time: {total_time} ms")
        print(f"gen len: {prev_pos - min_prompt_len}")
        print(f"average time: {total_time/(max_gen_len-1)}ms")

        if logprobs:
            token_logprobs = token_logprobs.tolist()
        out_tokens, out_logprobs = [], []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            probs = None
            if logprobs:
                probs = token_logprobs[i][start : len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any

            # if self.tokenizer.eos_token_id in toks:
                # eos_idx = toks.index(self.tokenizer.eos_token_id)
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
                probs = probs[:eos_idx] if logprobs else None
            out_tokens.append(toks)
            out_logprobs.append(probs)
        return (out_tokens, out_logprobs if logprobs else None)

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(x, bos=True, eos=False) for x in prompts]
        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
            echo=echo,
        )
        if generation_tokens == []:
            return []
        else:
            if logprobs:
                return [
                    {
                        "generation": self.tokenizer.decode(t),
                        "tokens": [self.tokenizer.decode(x) for x in t],
                        "logprobs": logprobs_i,
                    }
                    for t, logprobs_i in zip(generation_tokens, generation_logprobs)
                ]
            return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: List[Dialog],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        logprobs: bool = False,
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            unsafe_requests.append(
                any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
            )
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)

        generation_tokens, generation_logprobs = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            logprobs=logprobs,
        )
        if logprobs:
            return [
                {
                    "generation": {
                        "role": "assistant",
                        "content": self.tokenizer.decode(t)
                        if not unsafe
                        else UNSAFE_ERROR,
                    },
                    "tokens": [self.tokenizer.decode(x) for x in t],
                    "logprobs": logprobs_i,
                }
                for t, logprobs_i, unsafe in zip(
                    generation_tokens, generation_logprobs, unsafe_requests
                )
            ]
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t) if not unsafe else UNSAFE_ERROR,
                }
            }
            for t, unsafe in zip(generation_tokens, unsafe_requests)
        ]
