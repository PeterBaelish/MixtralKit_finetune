import torch
import time
import ctypes
from torch import nn
import threading

def copy_to_gpu(cpu_chunk, gpu_chunk):
    gpu_chunk.copy_(cpu_chunk)

def multi_threaded_cpu_to_gpu_transfer(gpu_tensor, cpu_tensor, num_threads, dim):

    cpu_chunks = torch.chunk(cpu_tensor, num_threads, dim=dim)
    gpu_chunks = torch.chunk(gpu_tensor, num_threads, dim=dim)

    threads = []
    for cpu_chunk, gpu_chunk in zip(cpu_chunks, gpu_chunks):
        thread = threading.Thread(target=copy_to_gpu, args=(cpu_chunk, gpu_chunk))
        threads.append(thread)

    # Starting threads
    for thread in threads:
        thread.start()

    # Joining threads
    for thread in threads:
        thread.join()

def copy_to_gpu_on_stream(cpu_chunk, gpu_chunk, stream, lib):
    rows = cpu_chunk.shape[0]
    cols = cpu_chunk.shape[1]

    src_cpu_memory_address = cpu_chunk.data_ptr()
    dst_gpu_memory_address = gpu_chunk.data_ptr()
    lib.copy2DTensorCpuToGpuOnStream(ctypes.c_void_p(dst_gpu_memory_address),
                ctypes.c_void_p(src_cpu_memory_address),
                ctypes.c_int(rows),
                ctypes.c_int(cols),
                stream)

def multi_threaded_cpu_to_gpu_transfer_on_stream(gpu_tensor, cpu_tensor, num_threads, dim, stream, lib):

    cpu_chunks = torch.chunk(cpu_tensor, num_threads, dim=dim)
    gpu_chunks = torch.chunk(gpu_tensor, num_threads, dim=dim)

    threads = []
    for cpu_chunk, gpu_chunk in zip(cpu_chunks, gpu_chunks):
        thread = threading.Thread(target=copy_to_gpu_on_stream, args=(cpu_chunk, gpu_chunk, stream, lib))
        threads.append(thread)

    # Starting threads
    for thread in threads:
        thread.start()

    # Joining threads
    for thread in threads:
        thread.join()

torch.set_printoptions(threshold=10005)

# Check if CUDA is available
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. This example requires a GPU.")

lib = ctypes.CDLL('/home/baelish/Desktop/MixtralKit_finetune/mixtralkit/layers/stream_manage.so')

lib.createStream.argtypes = []
lib.createStream.restype = ctypes.c_void_p

# copyCpuToGpuOnStream
lib.copyCpuToGpuOnStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.copyCpuToGpuOnStream.restype = None

 # copy2DTensorCpuToGpuOnStream
lib.copy2DTensorCpuToGpuOnStream.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.copy2DTensorCpuToGpuOnStream.restype = None

lib.copyCpuToGpuOnStream_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
lib.copyCpuToGpuOnStream_float.restype = None

 # copy2DTensorCpuToGpuOnStream
lib.copy2DTensorCpuToGpuOnStream_float.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
lib.copy2DTensorCpuToGpuOnStream_float.restype = None

# synchronizeStream
lib.synchronizeStream.argtypes = [ctypes.c_void_p]
lib.synchronizeStream.restype = None

# destroyStream
lib.destroyStream.argtypes = [ctypes.c_void_p]
lib.destroyStream.restype = None

# stream1 = lib.createStream()
stream1 = torch.cuda.Stream()
stream2 = lib.createStream()
stream3 = lib.createStream()

# Create two tensors
size = 4096
a = torch.full((1, size), 3.0, device='cuda')
c = torch.full((1, size), 1.0, device='cuda')
b = torch.full((size, 4*size), 2.0, device='cpu')
b_g = torch.full((size, 4*size), 3.0, device='cuda')
meta = torch.full((1, size), 1.0, device='cpu')


gate1 = nn.Linear(size, size*4, bias=False).cuda()
gate2 = nn.Linear(size*4, size, bias=False).cuda()

for _ in range(5):
    a = gate1(a)
    a = gate2(a)
torch.cuda.synchronize()

rows = b.shape[0]
cols = b.shape[1]
# It doesn't parallel when copy is in front of compute!!But parallel when compute is in front of copy!!

total_start_time = time.time()

for i in range(5):

    with torch.cuda.stream(stream1):
        num_threads = 4

        start_time_1 = time.time()
        
        multi_threaded_cpu_to_gpu_transfer_on_stream(b_g, b, 4, 0, stream3, lib)
        # multi_threaded_cpu_to_gpu_transfer(b_g, b, num_threads, 0)
        meta_g = meta
        meta_g.to("cuda")

        end_time_1 = time.time()
        elapsed_time_1 = (end_time_1 - start_time_1) * 1000
        print(f"in stream expert load time: {elapsed_time_1} ms")

    # Synchronize
    # lib.synchronizeStream(stream1)
    # lib.synchronizeStream(stream2)
    torch.cuda.synchronize()

    with torch.cuda.stream(stream1):

        #start_time = time.time()

        for _ in range(5):
            a = gate1(a)
            a = gate2(a)
        
        #end_time = time.time()
        #elapsed_time = (end_time - start_time) * 1000
        #print(f"compute time: {elapsed_time} ms")

    start_time = time.time()
    multi_threaded_cpu_to_gpu_transfer_on_stream(b_g, b, 4, 0, stream2, lib)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000
    print(f"df stream expert load time: {elapsed_time} ms")
    


torch.cuda.synchronize()
total_end_time = time.time()
elapsed_time = (total_end_time - total_start_time) * 1000
print(f"total time: {elapsed_time} ms")
print(b_g)


