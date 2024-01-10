# Copyright (c) OpenMMLab. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import json
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from .utils import ModelArgs
from .attention import TorchAttention, FairScaleAttention
from .ffn import TorchFFN, FairScaleFFN
from .transformer import TorchTransformerBlock, TorchTransformer, FairScaleTransformer
from hqq.core.quantize import *


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
        print("Softmax for Gate:{}".format(str(gate_softmax)))
        ''''''
        self.expert_gpu_w1 = nn.Linear(
            kwargs["dim"], kwargs["hidden_dim"], bias=False
        )
        self.expert_gpu_w2 = nn.Linear(
            kwargs["hidden_dim"], kwargs["dim"], bias=False
        )
        self.expert_gpu_w3 = nn.Linear(
            kwargs["dim"], kwargs["hidden_dim"], bias=False
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

        print("Selected experts", expert_indices)

        for i, expert in enumerate(self.experts):
            mask = (flat_expert_indices == i)
            if mask.any():
                start_time = time.time()

                num_threads = 4

                self.multi_threaded_cpu_to_gpu_transfer(self.expert_gpu_w1.W_q.data, expert.w1.W_q.data, num_threads, 0)
                self.multi_threaded_cpu_to_gpu_transfer(self.expert_gpu_w2.W_q.data, expert.w2.W_q.data, num_threads, 0)
                self.multi_threaded_cpu_to_gpu_transfer(self.expert_gpu_w3.W_q.data, expert.w3.W_q.data, num_threads, 0)

                #copy meta
                self.expert_gpu_w1.meta = {
                    key: value.to('cuda') if torch.is_tensor(value) else value
                    for key, value in expert.w1.meta.items()
                }
                self.expert_gpu_w2.meta = {
                    key: value.to('cuda') if torch.is_tensor(value) else value
                    for key, value in expert.w2.meta.items()
                }
                self.expert_gpu_w3.meta = {
                    key: value.to('cuda') if torch.is_tensor(value) else value
                    for key, value in expert.w3.meta.items()
                }

                '''
                self.ready        = False
                self.in_gpu       = False
                '''

                memory_stats = torch.cuda.memory_stats()
                print("current alloc mem GB:",memory_stats["allocated_bytes.all.current"]/(1024**3))

                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000
                print(f"expert copy time: {elapsed_time} ms")

                start_time = time.time()
                y[mask] = expert.w2(F.silu(expert.w1(x[mask])) * expert.w3(x[mask]))

                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000
                print(f"expert compute time: {elapsed_time} ms")
        
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