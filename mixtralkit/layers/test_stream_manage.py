import torch
import time
import ctypes
from torch import nn

torch.set_printoptions(threshold=10005)

# Check if CUDA is available
if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. This example requires a GPU.")

lib = ctypes.CDLL('/workspace/stream_manage.so')

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

# Create two tensors
size = 4096
a = torch.full((1, size), 3.0, device='cuda')
c = torch.full((1, size), 1.0, device='cuda')
b = torch.full((size, size), 2.0, device='cpu')
b_g = torch.full((size, size), 3.0, device='cuda')

gate1 = nn.Linear(size, size*4, bias=False).cuda()
gate2 = nn.Linear(size*4, size, bias=False).cuda()

torch.cuda.synchronize()

with torch.cuda.stream(stream1):
    for _ in range(5):
        a = gate1(a)
        a = gate2(a)

c = gate1(c)
c = gate2(c)

rows = b.shape[0]
cols = b.shape[1]

src_cpu_memory_address = b.data_ptr()
dst_gpu_memory_address = b_g.data_ptr()
lib.copy2DTensorCpuToGpuOnStream_float(ctypes.c_void_p(dst_gpu_memory_address),
            ctypes.c_void_p(src_cpu_memory_address),
            ctypes.c_int(rows),
            ctypes.c_int(cols),
            stream2)

# Synchronize
lib.synchronizeStream(stream1)
lib.synchronizeStream(stream2)
print(b_g)