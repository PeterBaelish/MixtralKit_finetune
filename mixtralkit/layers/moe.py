# Copyright (c) OpenMMLab. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import json
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import pycuda.driver as cuda_py
import pycuda.autoinit

import torch
import torch.nn.functional as F
from torch import nn
from .utils import ModelArgs
from .attention import TorchAttention, FairScaleAttention
from .ffn import TorchFFN_HQQ, TorchFFN, FairScaleFFN
from .transformer import TorchTransformerBlock, TorchTransformer, FairScaleTransformer
from .norm import RMSNorm
from .position_embeding import precompute_freqs_cis


class MoETorchFFN(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        num_shards: int,
        gate_softmax: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.experts = nn.ModuleList([
            TorchFFN(**kwargs).to(f"cuda:{i//num_shards}") 
            for i in range(num_experts)]
        )
        self.gate = nn.Linear(
            kwargs["dim"], num_experts, bias=False)
        
        self.num_experts_per_tok = num_experts_per_tok
        self.gate_softmax = gate_softmax
        print("Softmax for Gate:{}".format(str(gate_softmax)))

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        if self.gate_softmax:
            scores = self.gate(x).softmax(dim=-1)
        else:
            scores = self.gate(x)

        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)

        '''
        output_data = {
            "expert_indices": expert_indices.tolist()
            # "scores": scores.tolist()
        }

        with open("/workspace/MixtralKit/output_data.json", "a") as file:
            json.dump(output_data, file)
            file.write("\n")
        '''

        print("Selected experts", expert_indices)
        # print("scores of all experts", scores)

        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)

class SingleGPUMoETorchFFN(nn.Module):
    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        layer_id: int,
        gate_softmax: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.experts = nn.ModuleList([
            TorchFFN(**kwargs) for i in range(num_experts)]
        )
        self.gate = nn.Linear(
            kwargs["dim"], num_experts, bias=False)
        
        self.num_experts_per_tok = num_experts_per_tok
        self.gate_softmax = gate_softmax

        self.num_expert_cache = 2
        self.loaded_expert = [-1] * self.num_expert_cache

        print("Softmax for Gate:{}".format(str(gate_softmax)))

        # TODO: this should init on CPU, and load to GPU after quant
        #       We always have noneType here, don't know why
        self.experts_gpu = nn.ModuleList([
            TorchFFN_HQQ(**kwargs) for i in range(self.num_expert_cache)]
        )
        
        
    def copy_to_gpu(self, cpu_chunk, gpu_chunk):
        gpu_chunk.copy_(cpu_chunk)

    def multi_threaded_cpu_to_gpu_transfer(self, gpu_tensor, cpu_tensor, num_threads, dim):

        cpu_chunks = torch.chunk(cpu_tensor, num_threads, dim=dim)
        gpu_chunks = torch.chunk(gpu_tensor, num_threads, dim=dim)

        threads = []
        for cpu_chunk, gpu_chunk in zip(cpu_chunks, gpu_chunks):
            thread = threading.Thread(target=self.copy_to_gpu, args=(cpu_chunk, gpu_chunk))
            threads.append(thread)

        # Starting threads
        for thread in threads:
            thread.start()

        # Joining threads
        for thread in threads:
            thread.join()

    def load_expert_cpu_to_gpu(self, expert, gpu_expert, num_threads):
        start_time = time.time()

        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w1.W_q.data, expert.w1.W_q.data, num_threads, 0)
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w2.W_q.data, expert.w2.W_q.data, num_threads, 0)
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w3.W_q.data, expert.w3.W_q.data, num_threads, 0)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"expert weight copy time: {elapsed_time} ms")

        start_time = time.time()
        
        #copy meta
        self.experts_gpu[gpu_expert].w1.meta = {
            key: value.to('cuda') if torch.is_tensor(value) else value
            for key, value in expert.w1.meta.items()
        }
        self.experts_gpu[gpu_expert].w2.meta = {
            key: value.to('cuda') if torch.is_tensor(value) else value
            for key, value in expert.w2.meta.items()
        }
        self.experts_gpu[gpu_expert].w3.meta = {
            key: value.to('cuda') if torch.is_tensor(value) else value
            for key, value in expert.w3.meta.items()
        }
        
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        # print(f"expert meta copy time: {elapsed_time} ms")

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])
        device = x.device

        if self.gate_softmax:
            scores = self.gate(x).softmax(dim=-1)
        else:
            scores = self.gate(x)

        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)

        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        
        gpu_expert = 0

        print("Selected experts", expert_indices)

        for i, expert in enumerate(self.experts):
            mask = (flat_expert_indices == i)
            if mask.any():
            
                num_threads = 4

                if i not in self.loaded_expert:
                    if -1 in self.loaded_expert:
                        gpu_expert = self.loaded_expert.index(-1)
                    else:
                        gpu_expert = (gpu_expert + 1) % self.num_expert_cache
                        #TODO: LRU cache
                    self.load_expert_cpu_to_gpu(expert, gpu_expert, num_threads)
                    self.loaded_expert[gpu_expert] = i
                    print("Cache miss. copy expert ID:", i)
                else:
                    gpu_expert = self.loaded_expert.index(i)
                    print("Cache hit. hit expert ID:", i)

                
                # memory_stats = torch.cuda.memory_stats()
                # print("current alloc mem GB:",memory_stats["allocated_bytes.all.current"]/(1024**3))

                start_time = time.time()
                y[mask] = self.experts_gpu[gpu_expert](x[mask])

                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000
                # print(f"expert compute time: {elapsed_time} ms")
        
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)

class MoETorchTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        
        self.attention = TorchAttention(args)

        """
        assert args.moe["num_experts"] % args.num_gpus == 0, "num_experts must be divisible by num_gpus"
        self.feed_forward = MoETorchFFN(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_shards=args.moe["num_experts"] // args.num_gpus,
            **args.moe,
        )
        """

        self.feed_forward = SingleGPUMoETorchFFN(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            layer_id=layer_id,
            **args.moe,
        )

class PreloadMoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoETorchTransformerBlock(layer_id, params))
        
        least_priority, greatest_priority = cuda_py.Context.get_device().get_stream_priority_range()
        self.preload_stream = cuda_py.Stream(flags=cuda_py.stream_flags.NON_BLOCKING, priority=least_priority)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Perform a forward pass through the Transformer model.

        Args:
            tokens (torch.Tensor): Input token indices.
            start_pos (int): Starting position for attention caching.

        Returns:
            torch.Tensor: Output logits after applying the Transformer model.

        """
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            mask = torch.triu(mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for i, layer in enumerate(self.layers):

            next_feedforward = self.layers[i+1].feed_forward if i+1 < self.n_layers else None
            
            h = h + layer.attention.forward(
                layer.attention_norm(h), start_pos, freqs_cis, mask
            )
            
            if self.preload_stream is not None:
                self.preload_stream.synchronize()
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            with torch.cuda.stream(self.preload_stream):
                if next_feedforward is not None:
                    gpu_expert = 0

                    if start_pos == 0: #Prefill. We simply load expert 0 and 1, since it will use all of the expert mostly
                        flat_expert_indices = torch.tensor([0, 1])
                        for i, expert in enumerate(next_feedforward.experts):
                            expert_mask = (flat_expert_indices == i)
                            if expert_mask.any():
                                num_threads = 4
                                if i not in next_feedforward.loaded_expert:
                                    if -1 in next_feedforward.loaded_expert:
                                        gpu_expert = next_feedforward.loaded_expert.index(-1)
                                    else:
                                        gpu_expert = (gpu_expert + 1) % next_feedforward.num_expert_cache
                                    next_feedforward.load_expert_cpu_to_gpu(expert, gpu_expert, num_threads)
                                    next_feedforward.loaded_expert[gpu_expert] = i
                                else:
                                    gpu_expert = next_feedforward.loaded_expert.index(i)
                    else: # Decode
                        x = layer.ffn_norm(h)
                        x = x.view(-1, x.shape[-1])

                        if next_feedforward.gate_softmax:
                            scores = next_feedforward.gate(x).softmax(dim=-1)
                        else:
                            scores = next_feedforward.gate(x)

                        expert_weights, expert_indices = torch.topk(
                            scores, next_feedforward.num_experts_per_tok, dim=-1)
                        
                        flat_expert_indices = expert_indices.view(-1)

                        print("Predict experts", expert_indices)
                        for i, expert in enumerate(next_feedforward.experts):
                            expert_mask = (flat_expert_indices == i)
                            if expert_mask.any():
                                num_threads = 4
                                if i not in next_feedforward.loaded_expert:
                                    if -1 in next_feedforward.loaded_expert:
                                        gpu_expert = next_feedforward.loaded_expert.index(-1)
                                    else:
                                        gpu_expert = (gpu_expert + 1) % next_feedforward.num_expert_cache
                                        #TODO: LRU cache
                                    next_feedforward.load_expert_cpu_to_gpu(expert, gpu_expert, num_threads)
                                    next_feedforward.loaded_expert[gpu_expert] = i
                                else:
                                    gpu_expert = next_feedforward.loaded_expert.index(i)

            h = h + layer.feed_forward.forward(layer.ffn_norm(h))
        
        h = self.norm(h)
        output = self.output(h).float()
        return output


class MoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoETorchTransformerBlock(layer_id, params))

"""
Implementation for FairScale Backend
TODO: Stay Tuned.
"""

class MoEFairScaleFFN(nn.Module):
    def __init__(self,
                 num_experts: int,
                 num_experts_per_tok: int,
                 **kwargs):
        super().__init__()
        from fairscale.nn.model_parallel.layers import (
            ColumnParallelLinear,
        )
        self.experts = nn.ModuleList(
            [FairScaleFFN(**kwargs) for i in range(num_experts)]
        )
        self.gate = ColumnParallelLinear(
            kwargs["dim"], num_experts, bias=False, init_method=lambda x: x
        )        
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x):
        orig_shape = x.shape
        x = x.view(-1, x.shape[-1])

        scores = self.gate(x)
        expert_weights, expert_indices = torch.topk(
            scores, self.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)
        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)



class MoEFairScaleTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        self.attention = FairScaleAttention(args)
        self.feed_forward = MoEFairScaleFFN(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            **args.moe
        )


class MoEFairScaleTransformer(FairScaleTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoETorchTransformerBlock(layer_id, params))