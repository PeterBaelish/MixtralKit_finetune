import torch
import ctypes
import time
import threading
import matplotlib.pyplot as plt
import numpy as np

#torch.set_printoptions(threshold=100000000)

# 加载动态链接库
lib = ctypes.CDLL('./libtensorcompress.so')

# 设置C++函数的参数类型
lib.compressTensor.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int
]

def copy_to_gpu(cpu_chunk, gpu_chunk):
    gpu_chunk.copy_(cpu_chunk)

def multi_threaded_cpu_to_gpu_transfer(gpu_tensor, cpu_tensor, num_threads, dim):

    cpu_chunks = torch.chunk(cpu_tensor, num_threads, dim=dim)
    gpu_chunks = torch.chunk(gpu_tensor, num_threads, dim=dim)

    threads = []
    for cpu_chunk, gpu_chunk in zip(cpu_chunks, gpu_chunks):
        thread = threading.Thread(target=copy_to_gpu, args=(cpu_chunk, gpu_chunk)) # TODO: multiprocessing
        threads.append(thread)

    # Starting threads
    for thread in threads:
        thread.start()

    # Joining threads
    for thread in threads:
        thread.join()

# 设置C++函数的返回类型
lib.compressTensor.restype = None

# 创建PyTorch张量
l = 1408
h = 16

compress_line = []
copy_line = []
decompress_line = []
total_line = []

for i in range(23):

    print("============================test sparsity:", i*64)
    compress_time = []
    copy_time = []
    decompress_time = []

    for j in range(10):

        A = torch.randint(low=0, high=256, size=(l, h*64), dtype=torch.uint8)
        B = torch.randint(low=0, high=256, size=(l, h*64), dtype=torch.uint8, device="cuda")

        mask = torch.full((l,), False, dtype=torch.bool)
        N = i*64
        indices = torch.randperm(l)[:N]
        mask[indices] = True
        mask = mask.squeeze()

        print(mask)
        gpu_mask = mask.to("cuda")

        sparse_size = torch.sum(mask).item()

        ####################################################################################compress

        start_time = time.time()

        sparse_cpu_tensor = A[mask]

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"PYTHON compress time: {elapsed_time} ms")

        mask = torch.nonzero(mask, as_tuple=True)[0]

        # 获取张量的指针
        A_ptr = A.data_ptr()
        mask_ptr = mask.data_ptr()

        # 定义线程数量
        num_threads = 2

        output = torch.empty((sparse_size, h*64), dtype=torch.uint8)
        output_ptr = output.data_ptr()

        start_time = time.time()
        # 调用C++函数
        lib.compressTensor(ctypes.c_void_p(A_ptr), ctypes.c_void_p(mask_ptr), ctypes.c_void_p(output_ptr), ctypes.c_int(l), ctypes.c_int(h*64), ctypes.c_int(sparse_size), ctypes.c_int(num_threads))
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"C++ compress time: {elapsed_time} ms")
        compress_time.append(elapsed_time)

        print(torch.allclose(sparse_cpu_tensor, output, atol=1e-05))

        ##########################################################################################copy

        output_gpu = torch.empty((sparse_size, h*64), dtype=torch.uint8, device="cuda")

        start_time = time.time()

        multi_threaded_cpu_to_gpu_transfer(output_gpu, output, num_threads, 0)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"copy time: {elapsed_time} ms")
        copy_time.append(elapsed_time)

        ##########################################################################################decompress

        start_time = time.time()

        B[gpu_mask] = output_gpu

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000
        print(f"decompress time: {elapsed_time} ms")
        decompress_time.append(elapsed_time)
    
    decompress_average = sum(decompress_time) / len(decompress_time)
    copy_average = sum(copy_time) / len(copy_time)
    compress_average = sum(compress_time) / len(compress_time)
    
    compress_line.append(compress_average)
    copy_line.append(copy_average)
    decompress_line.append(decompress_average)
    total_line.append(decompress_average+copy_average+compress_average)

print("compress_line:", compress_line)
print("copy_line:", copy_line)
print("decompress_line:", decompress_line)
print("total_line:", total_line)

C=A
start_time = time.time()
B = C.to(B.device)
end_time = time.time()
elapsed_time = (end_time - start_time) * 1000
print(f"origin copy time: {elapsed_time} ms")

# 创建一个图形和一个坐标轴
fig, ax = plt.subplots()

x = [i * 64 for i in range(23)]

# 绘制每个列表作为一条线
ax.plot(x, compress_line, label='Compress latency')
ax.plot(x, copy_line, label='Load latency')
ax.plot(x, decompress_line, label='Decompress latency')
ax.plot(x, total_line, label='Total latency')

x_ticks = [x[0], x[6], x[11], x[16], x[22]]
ax.set_xticks(x_ticks)

# 添加一些图形的装饰
ax.set_xlabel('Number of active neurons')  # X轴标签
ax.set_ylabel('Latency(ms)')  # Y轴标签
ax.set_title('Latency breakdown of neuron-sparse loading')  # 图形标题
ax.legend()  # 添加图例


fig.tight_layout()

# 保存图像
plt.savefig('bar_chart.png')
plt.close()  # 关闭图表以释放内存