// nvcc -shared -o stream_manage.so stream_manage.cu -L/usr/local/cuda/lib64 -lcudart

extern "C" cudaStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}

extern "C" void copyCpuToGpuOnStream(int8_t *dst, const int8_t *src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(int8_t), cudaMemcpyHostToDevice, stream);
}

extern "C" void copy2DTensorCpuToGpuOnStream(int8_t *dst, const int8_t *src, int rows, int cols, cudaStream_t stream) {
    size_t size = rows * cols * sizeof(int8_t);
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}


extern "C" void synchronizeStream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

extern "C" void destroyStream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}
