# Copyright (c) OpenMMLab. and affiliates.
# Copyright (c) Meta Platforms, Inc. and affiliates.

import math
import json
import time
import threading
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import ctypes

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

        # print("Selected experts", expert_indices)
        # print("scores of all experts", scores)

        flat_expert_indices = expert_indices.view(-1)

        x = x.repeat_interleave(self.num_experts_per_tok, dim=0)
        y = torch.empty_like(x)
        for i, expert in enumerate(self.experts):
            y[flat_expert_indices == i] = expert(x[flat_expert_indices == i])
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)

class QuantMoETorchFFN(nn.Module):
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
            TorchFFN_HQQ(**kwargs) for i in range(num_experts)]
        )
        self.gate = nn.Linear(
            kwargs["dim"], num_experts, bias=False)
        
        self.num_experts_per_tok = num_experts_per_tok
        self.gate_softmax = gate_softmax
        print("Softmax for Gate:{}".format(str(gate_softmax)))

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
        
        # print("Selected experts", expert_indices)
        '''
        output_data = {
            "expert_indices": expert_indices.tolist()
            # "scores": scores.tolist()
        }
        
        #if len(expert_indices.tolist()) > 1:
        with open("/workspace/MixtralKit/output_data.json", "a") as file:
            json.dump(output_data, file)
            file.write("\n")
        '''

        for i, expert in enumerate(self.experts):
            mask = (flat_expert_indices == i)
            if mask.any():

                start_time = time.time()
                y[mask] = expert(x[mask])

                end_time = time.time()
                elapsed_time = (end_time - start_time) * 1000
                # print(f"expert compute time: {elapsed_time} ms")
        
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)

class SparsePredict(nn.Module):
    def __init__(
        self,
        dim: int,
        predict_hidden_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        # SparsePredict store on GPU since it only have ~1G
        self.w1 = nn.Linear(
            dim, predict_hidden_dim, bias=False
        )
        self.w2 = nn.Linear(
            predict_hidden_dim, hidden_dim, bias=False
        )

    def forward(self, x):
        return self.w2(self.w1(x))


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

        self.experts_gpu = nn.ModuleList([
            TorchFFN_HQQ(**kwargs) for i in range(self.num_expert_cache)]
        )
        
        # MLP to predict expert sparsity
        # NOTICE: We actually predict the sparsity of SiLU output. since SiLU is not 0 when input is 0, so we need a threshold
        # output: sparse is 0, activate is 1
        self.predict_hidden_dim = 256
        self.sparse_predict = nn.ModuleList([
            SparsePredict(kwargs["dim"], self.predict_hidden_dim, kwargs["hidden_dim"]) for i in range(num_experts)]
        )
        
    def copy_to_gpu(self, cpu_chunk, gpu_chunk):
        gpu_chunk.copy_(cpu_chunk)
    
    def copy_to_gpu_on_stream(self, cpu_chunk, gpu_chunk, stream, lib):
        rows = cpu_chunk.shape[0]
        cols = cpu_chunk.shape[1]

        src_cpu_memory_address = cpu_chunk.data_ptr()
        dst_gpu_memory_address = gpu_chunk.data_ptr()
        lib.copy2DTensorCpuToGpuOnStream(ctypes.c_void_p(dst_gpu_memory_address),
                 ctypes.c_void_p(src_cpu_memory_address),
                 ctypes.c_int(rows),
                 ctypes.c_int(cols),
                 stream)

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
    
    def multi_threaded_cpu_to_gpu_transfer_on_stream(self, gpu_tensor, cpu_tensor, num_threads, dim, stream, lib):

        cpu_chunks = torch.chunk(cpu_tensor, num_threads, dim=dim)
        gpu_chunks = torch.chunk(gpu_tensor, num_threads, dim=dim)

        threads = []
        for cpu_chunk, gpu_chunk in zip(cpu_chunks, gpu_chunks):
            thread = threading.Thread(target=self.copy_to_gpu_on_stream, args=(cpu_chunk, gpu_chunk, stream, lib))
            threads.append(thread)

        # Starting threads
        for thread in threads:
            thread.start()

        # Joining threads
        for thread in threads:
            thread.join()

    def sparse_multi_threaded_cpu_to_gpu_transfer_on_stream(self, y, meta, gpu_tensor, cpu_tensor, num_threads, dim, stream, lib):
        
        # 0 maybe not competitate with HQQquant. We need to modify weight to align with meta data, after meta data, the weight should be 0
        
        zero_quant = meta['zero'].to(torch.uint8)

        real_mask = y.squeeze() == 1 # (hidden_dim)
        mask = torch.tensor([real_mask[2*i] or real_mask[2*i+1] for i in range(real_mask.size(0) // 2)])
        # HQQ quant in 4bit but store in 8bit, so if load in ? we need to load ?
        # it is not equal with w2 and w1,w3, w2 don't need zero_quant

        real_sparse_size = torch.sum(real_mask).item()
        sparse_size = torch.sum(mask).item()
        
        # sparse on w1,w3 cols and w2 rows
        if cpu_tensor.shape[0] > cpu_tensor.shape[1]: # w2
            sparse_cpu_tensor = cpu_tensor[real_mask]
            sparse_gpu_tensor = gpu_tensor[:real_sparse_size]
        else: # w1, w3
            sparse_cpu_tensor = cpu_tensor[:, mask]
            sparse_gpu_tensor = gpu_tensor[:, :sparse_size]
        
        # copy
        sparse_cpu_chunks = torch.chunk(sparse_cpu_tensor, num_threads, dim=dim)
        sparse_gpu_chunks = torch.chunk(sparse_gpu_tensor, num_threads, dim=dim)

        threads = []
        for cpu_chunk, gpu_chunk in zip(sparse_cpu_chunks, sparse_gpu_chunks):
            thread = threading.Thread(target=self.copy_to_gpu_on_stream, args=(cpu_chunk, gpu_chunk, stream, lib))
            threads.append(thread)

        # Starting threads
        for thread in threads:
            thread.start()

        # Joining threads
        for thread in threads:
            thread.join()
        
        # reorder weight
        if cpu_tensor.shape[0] > cpu_tensor.shape[1]:  # w2
            j = 0
            for i in range(real_mask.size(0)):
                if real_mask[i]:
                    gpu_tensor[i] = sparse_gpu_tensor[j]
                    j += 1
                else:
                    gpu_tensor[i] = zero_quant
        else:  # w1, w3
            j = 0
            for i in range(mask.size(0)):
                if mask[i]:
                    gpu_tensor[:, i] = sparse_gpu_tensor[:, j]
                    j += 1
                    if real_mask[2*i] == False:
                        gpu_tensor[:, i] = (zero_quant << 4) | gpu_tensor[:, i]
                    if real_mask[2*i+1] == False:
                        gpu_tensor[:, i] = gpu_tensor[:, i] | zero_quant
                else:
                    gpu_tensor[:, i] = (zero_quant << 4) | zero_quant
                
    def load_expert_cpu_to_gpu(self, expert, gpu_expert, num_threads):
        # start_time = time.time()
        '''
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w1.W_q.data, expert.w1.W_q.data, num_threads, 0)
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w2.W_q.data, expert.w2.W_q.data, num_threads, 0)
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w3.W_q.data, expert.w3.W_q.data, num_threads, 0)
        '''

        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w1.weight.data, expert.w1.weight.data, num_threads, 0)
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w2.weight.data, expert.w2.weight.data, num_threads, 0)
        self.multi_threaded_cpu_to_gpu_transfer(self.experts_gpu[gpu_expert].w3.weight.data, expert.w3.weight.data, num_threads, 0)

        # end_time = time.time()
        # elapsed_time = (end_time - start_time) * 1000
        # print(f"expert weight copy time: {elapsed_time} ms")

        # start_time = time.time()
        '''
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
        '''
        # end_time = time.time()
        # elapsed_time = (end_time - start_time) * 1000
        # print(f"expert meta copy time: {elapsed_time} ms")

    def load_expert_cpu_to_gpu_on_stream(self, expert, gpu_expert, num_threads, stream, lib):
        # start_time = time.time()

        self.multi_threaded_cpu_to_gpu_transfer_on_stream(self.experts_gpu[gpu_expert].w1.W_q.data, expert.w1.W_q.data, num_threads, 0, stream, lib)
        self.multi_threaded_cpu_to_gpu_transfer_on_stream(self.experts_gpu[gpu_expert].w2.W_q.data, expert.w2.W_q.data, num_threads, 0, stream, lib)
        self.multi_threaded_cpu_to_gpu_transfer_on_stream(self.experts_gpu[gpu_expert].w3.W_q.data, expert.w3.W_q.data, num_threads, 0, stream, lib)

        # end_time = time.time()
        # elapsed_time = (end_time - start_time) * 1000
        # print(f"expert weight copy time: {elapsed_time} ms")

        # start_time = time.time()
        
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
        
        # end_time = time.time()
        # elapsed_time = (end_time - start_time) * 1000
        # print(f"expert meta copy time: {elapsed_time} ms")
    
    def sparse_load_expert_cpu_to_gpu_on_stream(self, x, expert, gpu_expert, num_threads, stream, lib):
        
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

        # TODO: actually, we shouldn't predict here
        y = self.sparse_predict(x[0]) #(1, dim) -> (1, hidden_dim)
        # TODO: for testing , we need to random y as 80% to 0, 20% to 1

        self.sparse_multi_threaded_cpu_to_gpu_transfer_on_stream(y, self.experts_gpu[gpu_expert].w1.meta, self.experts_gpu[gpu_expert].w1.W_q.data, expert.w1.W_q.data, num_threads, 0, stream, lib)
        self.sparse_multi_threaded_cpu_to_gpu_transfer_on_stream(y, self.experts_gpu[gpu_expert].w2.meta, self.experts_gpu[gpu_expert].w2.W_q.data, expert.w2.W_q.data, num_threads, 0, stream, lib)
        self.sparse_multi_threaded_cpu_to_gpu_transfer_on_stream(y, self.experts_gpu[gpu_expert].w3.meta, self.experts_gpu[gpu_expert].w3.W_q.data, expert.w3.W_q.data, num_threads, 0, stream, lib)

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

        # print("Selected experts", expert_indices)

        for i, expert in enumerate(self.experts):
            mask = (flat_expert_indices == i)
            if mask.any():
            
                num_threads = 4

                if i not in self.loaded_expert:
                    if -1 in self.loaded_expert:
                        gpu_expert = self.loaded_expert.index(-1)
                    else:
                        gpu_expert = (gpu_expert + 1) % self.num_expert_cache

                    self.load_expert_cpu_to_gpu(expert, gpu_expert, num_threads)
                    self.loaded_expert[gpu_expert] = i
                    # print("Cache miss. copy expert ID:", i)
                else:
                    gpu_expert = self.loaded_expert.index(i)
                    # print("Cache hit. hit expert ID:", i)

                
                # memory_stats = torch.cuda.memory_stats()
                # print("current alloc mem GB:",memory_stats["allocated_bytes.all.current"]/(1024**3))

                # start_time = time.time()
                y[mask] = self.experts_gpu[gpu_expert](x[mask])

                # end_time = time.time()
                # elapsed_time = (end_time - start_time) * 1000
                # print(f"expert compute time: {elapsed_time} ms")
        
        y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
        return y.view(*orig_shape)

class MoETorchTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        
        self.self_attn = TorchAttention(args)

        assert args.moe["num_experts"] % args.num_gpus == 0, "num_experts must be divisible by num_gpus"
        self.block_sparse_moe = MoETorchFFN(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_shards=args.moe["num_experts"] // args.num_gpus,
            **args.moe,
        )

class QuantMoETorchTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        
        self.self_attn = TorchAttention(args)

        assert args.moe["num_experts"] % args.num_gpus == 0, "num_experts must be divisible by num_gpus"
        self.block_sparse_moe = QuantMoETorchFFN(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            num_shards=args.moe["num_experts"] // args.num_gpus,
            **args.moe,
        )

class SingleGPUMoETorchTransformerBlock(TorchTransformerBlock):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__(layer_id, args)
        
        self.self_attn = TorchAttention(args)

        self.block_sparse_moe = SingleGPUMoETorchFFN(
            dim=args.dim,
            hidden_dim=args.hidden_dim,
            layer_id=layer_id,
            **args.moe,
        )

class SingleGPUMoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(SingleGPUMoETorchTransformerBlock(layer_id, params))

class PreloadMoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(SingleGPUMoETorchTransformerBlock(layer_id, params))
        
        self.lib = ctypes.CDLL('/home/taoziyang/MixtralKit_finetune/mixtralkit/layers/stream_manage.so')

        self.lib.createStream.argtypes = []
        self.lib.createStream.restype = ctypes.c_void_p

        #TODO: pinned memory? Can we design a CPU mem control that we can accelerate the copy time

        # copyCpuToGpuOnStream
        self.lib.copyCpuToGpuOnStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.copyCpuToGpuOnStream.restype = None

        # copy2DTensorCpuToGpuOnStream
        self.lib.copy2DTensorCpuToGpuOnStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        self.lib.copy2DTensorCpuToGpuOnStream.restype = None

        # synchronizeStream
        self.lib.synchronizeStream.argtypes = [ctypes.c_void_p]
        self.lib.synchronizeStream.restype = None

        # destroyStream
        self.lib.destroyStream.argtypes = [ctypes.c_void_p]
        self.lib.destroyStream.restype = None

        self.stream = self.lib.createStream()

        self.normal_stream = torch.cuda.Stream()

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
        h = self.embed_tokens(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        attn_mask = None
        if seqlen > 1:
            attn_mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            attn_mask = torch.triu(attn_mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            attn_mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                attn_mask
            ]).type_as(h)


        # code before layer 0 expert copy
        # Layer0 Attn
        h = h + self.layers[0].self_attn.forward(
            self.layers[0].input_layernorm(h), start_pos, freqs_cis, attn_mask
        )
        h_store = h

        # Gate for layer0 expert
        z = self.layers[0].post_attention_layernorm(h)

        orig_shape = z.shape
        z = z.view(-1, z.shape[-1])

        if self.layers[0].block_sparse_moe.gate_softmax:
            scores = self.layers[0].block_sparse_moe.gate(z).softmax(dim=-1)
        else:
            scores = self.layers[0].block_sparse_moe.gate(z)

        expert_weights, expert_indices = torch.topk(
            scores, self.layers[0].block_sparse_moe.num_experts_per_tok, dim=-1)
        expert_weights = expert_weights.softmax(dim=-1)

        flat_expert_indices = expert_indices.view(-1)

        z = z.repeat_interleave(self.layers[0].block_sparse_moe.num_experts_per_tok, dim=0)
        y = torch.empty_like(z)

        # print("Selected experts", expert_indices)

        # predict layer1 expert
        
        x = self.layers[1].post_attention_layernorm(h_store)
        x = x.view(-1, x.shape[-1])

        if self.layers[1].block_sparse_moe.gate_softmax:
            scores = self.layers[1].block_sparse_moe.gate(x).softmax(dim=-1)
        else:
            scores = self.layers[1].block_sparse_moe.gate(x)

        predict_expert_weights, predict_expert_indices = torch.topk(
            scores, self.layers[1].block_sparse_moe.num_experts_per_tok, dim=-1)
        
        predict_flat_expert_indices = predict_expert_indices.view(-1)
        # print("Predict experts", predict_expert_indices)

        # print("token begin")
        for i, layer in enumerate(self.layers):

            # Copy expert for i layer
            with torch.cuda.stream(self.normal_stream):

                preload_flat_expert_indices = predict_flat_expert_indices
                gpu_expert = 0

                #split copy and compute in decode, but normal in prefill
                if start_pos == 0: # prefill. We don't split copy and compute here since we cannot load all the experts in the same time

                    for j, expert in enumerate(layer.block_sparse_moe.experts):
                        mask = (flat_expert_indices == j)
                        if mask.any():
                        
                            num_threads = 4

                            if j not in layer.block_sparse_moe.loaded_expert:
                                if -1 in layer.block_sparse_moe.loaded_expert:
                                    gpu_expert = layer.block_sparse_moe.loaded_expert.index(-1)
                                else:
                                    gpu_expert = (gpu_expert + 1) % layer.block_sparse_moe.num_expert_cache

                                layer.block_sparse_moe.load_expert_cpu_to_gpu(expert, gpu_expert, num_threads)
                                layer.block_sparse_moe.loaded_expert[gpu_expert] = j
                                # print("Cache miss. copy expert ID:", j)
                            else:
                                gpu_expert = layer.block_sparse_moe.loaded_expert.index(j)
                                # print("Cache hit. hit expert ID:", j)

                            
                            # memory_stats = torch.cuda.memory_stats()
                            # print("current alloc mem GB:",memory_stats["allocated_bytes.all.current"]/(1024**3))

                            # start_time = time.time()
                            y[mask] = layer.block_sparse_moe.experts_gpu[gpu_expert](z[mask])

                            # end_time = time.time()
                            # elapsed_time = (end_time - start_time) * 1000
                            # print(f"expert compute time: {elapsed_time} ms")
                
                else: #decode. We split compute and copy and only copy here to make more parallel between compute and preload
                    
                    for j, expert in enumerate(layer.block_sparse_moe.experts):
                        mask = (flat_expert_indices == j)
                        if mask.any():
                        
                            num_threads = 4

                            if j not in layer.block_sparse_moe.loaded_expert:
                                if -1 in layer.block_sparse_moe.loaded_expert:
                                    gpu_expert = layer.block_sparse_moe.loaded_expert.index(-1)
                                else:
                                    gpu_expert = (gpu_expert + 1) % layer.block_sparse_moe.num_expert_cache

                                # start_time = time.time()
                                layer.block_sparse_moe.load_expert_cpu_to_gpu(expert, gpu_expert, num_threads)
                                # end_time = time.time()

                                # elapsed_time = (end_time - start_time) * 1000
                                # print(f"expert load time: {elapsed_time} ms")

                                layer.block_sparse_moe.loaded_expert[gpu_expert] = j
                                # print("Cache miss. copy expert ID:", j)
                            else:
                                gpu_expert = layer.block_sparse_moe.loaded_expert.index(j)
                                # print("Cache hit. hit expert ID:", j)
                            # memory_stats = torch.cuda.memory_stats()
                            # print("current alloc mem GB:",memory_stats["allocated_bytes.all.current"]/(1024**3))

            # Sync
            # self.lib.synchronizeStream(self.stream)
            self.lib.synchronizeStream(self.normal_stream)

            ## VERY IMPORTANT: COMPUTE MUST BE IN FRONT OF COPY, OTHER WISE THEY WON'T PARALLEL!!   

            with torch.cuda.stream(self.normal_stream):

                # normal i MoEFFN when Decode
                if start_pos != 0:
                    for j, expert in enumerate(layer.block_sparse_moe.experts):
                        mask = (flat_expert_indices == j)
                        if mask.any():
                            gpu_expert = layer.block_sparse_moe.loaded_expert.index(j)

                            # start_time = time.time()
                            y[mask] = layer.block_sparse_moe.experts_gpu[gpu_expert](z[mask])
                            # end_time = time.time()
                            # elapsed_time = (end_time - start_time) * 1000
                            # print(f"expert compute time: {elapsed_time} ms")
                   
                y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
                y = y.view(*orig_shape)
                h = y + h_store

                #normal i+1 Attn
                next_attention = self.layers[i+1].self_attn if i+1 < self.n_layers else None

                if next_attention is not None:
                    h = h + next_attention.forward(
                        self.layers[i+1].input_layernorm(h), start_pos, freqs_cis, attn_mask
                    )
                    h_store = h

                # normal Gate for i+1 layer expert
                next_feedforward = self.layers[i+1].block_sparse_moe if i+1 < self.n_layers else None

                if next_feedforward is not None:

                    z = self.layers[i+1].post_attention_layernorm(h_store)

                    orig_shape = z.shape
                    z = z.view(-1, z.shape[-1])

                    if next_feedforward.gate_softmax:
                        scores = next_feedforward.gate(z).softmax(dim=-1)
                    else:
                        scores = next_feedforward.gate(z)

                    expert_weights, expert_indices = torch.topk(
                        scores, next_feedforward.num_experts_per_tok, dim=-1)
                    expert_weights = expert_weights.softmax(dim=-1)

                    flat_expert_indices = expert_indices.view(-1)

                    z = z.repeat_interleave(next_feedforward.num_experts_per_tok, dim=0)
                    y = torch.empty_like(z)

                    # print("Selected experts", expert_indices)

                # predict i+2 expert
                predict_feedforward = self.layers[i+2].block_sparse_moe if i+2 < self.n_layers else None

                if predict_feedforward is not None:
                    x = self.layers[i+2].post_attention_layernorm(h_store)
                    x = x.view(-1, x.shape[-1])

                    if predict_feedforward.gate_softmax:
                        scores = predict_feedforward.gate(x).softmax(dim=-1)
                    else:
                        scores = predict_feedforward.gate(x)

                    predict_expert_weights, predict_expert_indices = torch.topk(
                        scores, predict_feedforward.num_experts_per_tok, dim=-1)
                    
                    predict_flat_expert_indices = predict_expert_indices.view(-1)
                    # print("Predict experts", predict_expert_indices)

            # Preload i+1 expert
            preload_feedforward = self.layers[i+1].block_sparse_moe if i+1 < self.n_layers else None

            if preload_feedforward is not None:
                pre_gpu_expert = 0

                if start_pos == 0: #Prefill. We simply load expert 0 and 1, since it will use all of the expert mostly

                    preload_flat_expert_indices = torch.tensor([0, 1])

                    for j, expert in enumerate(preload_feedforward.experts):
                        expert_mask = (preload_flat_expert_indices == j)
                        if expert_mask.any():
                            num_threads = 4
                            if j not in preload_feedforward.loaded_expert:
                                if -1 in preload_feedforward.loaded_expert:
                                    pre_gpu_expert = preload_feedforward.loaded_expert.index(-1)
                                else:
                                    pre_gpu_expert = (pre_gpu_expert + 1) % preload_feedforward.num_expert_cache
                                preload_feedforward.load_expert_cpu_to_gpu_on_stream(expert, pre_gpu_expert, num_threads, self.stream, self.lib)
                                preload_feedforward.loaded_expert[pre_gpu_expert] = j
                            else:
                                pre_gpu_expert = preload_feedforward.loaded_expert.index(j)
                else: # Decode

                    for j, expert in enumerate(preload_feedforward.experts):
                        expert_mask = (preload_flat_expert_indices == j)
                        if expert_mask.any():
                            num_threads = 4
                            if j not in preload_feedforward.loaded_expert:
                                if -1 in preload_feedforward.loaded_expert:
                                    pre_gpu_expert = preload_feedforward.loaded_expert.index(-1)
                                else:
                                    pre_gpu_expert = (pre_gpu_expert + 1) % preload_feedforward.num_expert_cache

                                # start_time = time.time()
                                preload_feedforward.load_expert_cpu_to_gpu_on_stream(expert, pre_gpu_expert, num_threads, self.stream, self.lib)
                                # end_time = time.time()
                                # elapsed_time = (end_time - start_time) * 1000
                                # print(f"expert preload time: {elapsed_time} ms")

                                preload_feedforward.loaded_expert[pre_gpu_expert] = j
                            else:
                                pre_gpu_expert = preload_feedforward.loaded_expert.index(j)

        torch.cuda.synchronize()
        
        h = self.norm(h)
        output = self.lm_head(h).float()
        return output

class QuantMoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(QuantMoETorchTransformerBlock(layer_id, params))
    ''''''
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
        h = self.embed_tokens(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]

        attn_mask = None
        if seqlen > 1:
            attn_mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=tokens.device
            )

            attn_mask = torch.triu(attn_mask, diagonal=1)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            attn_mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                attn_mask
            ]).type_as(h)

        output_file_path = "/workspace/MixtralKit_finetune/output_data.json"

        layer_id = 0
        for layer in self.layers:

            # we may output here
            ''''''
            if start_pos != 0:
                input_tensor = h.tolist()
                # print("input_tensor: ", input_tensor)
                with open(output_file_path, 'a') as file:
                    file.write(str(input_tensor))
            
            h = h + layer.self_attn.forward(
                layer.input_layernorm(h), start_pos, freqs_cis, attn_mask
            )

            h_store = h # output
            
            '''
            if start_pos != 0:
                input_tensor = h.tolist()
                print(input_tensor)
                with open(output_file_path, 'a') as file:
                    file.write(str(input_tensor))
            '''
            
            h = layer.post_attention_layernorm(h) 

            orig_shape = h.shape
            h = h.view(-1, h.shape[-1])

            if layer.block_sparse_moe.gate_softmax:
                scores = layer.block_sparse_moe.gate(h).softmax(dim=-1)
            else:
                scores = layer.block_sparse_moe.gate(h)

            expert_weights, expert_indices = torch.topk(
                scores, layer.block_sparse_moe.num_experts_per_tok, dim=-1)
            expert_weights = expert_weights.softmax(dim=-1)

            flat_expert_indices = expert_indices.view(-1)

            h = h.repeat_interleave(layer.block_sparse_moe.num_experts_per_tok, dim=0)
            y = torch.empty_like(h)

            for i, expert in enumerate(layer.block_sparse_moe.experts):
                mask = (flat_expert_indices == i)
                if mask.any():

                    # y[mask] = expert(h[mask])
                    h_m = h[mask]

                    device = h_m.device
                    h_m = h_m.to(expert.w1.W_q.device)

                    h_m_store = h_m
                    h_m = F.silu(expert.w1(h_m_store)) # output

                    if start_pos != 0:
                        threshold = 0.1
                        sparsity = torch.where(torch.abs(h_m) < threshold, torch.tensor(0, device=h_m.device), torch.tensor(1, device=h_m.device))
                        sparsity_list = sparsity.tolist()
                        print("sparsity_list: ", sparsity_list)
                        with open(output_file_path, 'a') as file:
                            file.write(str(sparsity_list))
                    
                    h_m = h_m * expert.w3(h_m_store)
                    y[mask] = expert.w2(h_m).to(device)

            y = (y.view(*expert_weights.shape, -1) * expert_weights.unsqueeze(-1)).sum(dim=1)
            h = y.view(*orig_shape)

            h = h_store + h

            layer_id = layer_id + 1
        
        h = self.norm(h)
        output = self.lm_head(h).float()
        return output
    

class MoETorchTransformer(TorchTransformer):
    def __init__(self, params: ModelArgs):
        super().__init__(params)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(MoETorchTransformerBlock(layer_id, params))

"""
Implementation for FairScale Backend
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
        self.self_attn = FairScaleAttention(args)
        self.block_sparse_moe = MoEFairScaleFFN(
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