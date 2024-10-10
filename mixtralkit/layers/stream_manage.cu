// compile for python
// nvcc -shared -o stream_manage.so stream_manage.cu -L/usr/local/cuda/lib64 -lcudart

// compile for C++
// nvcc add_floats.c stream_manage.cu -o stream_manage -L/usr/local/cuda/lib64 -lcudart
// ./stream_manage

#include <stdio.h>
#include "add_floats.h"

extern "C" cudaStream_t createStream() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return stream;
}

extern "C" void copyCpuToGpuOnStream(int8_t *dst, const int8_t *src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(int8_t), cudaMemcpyHostToDevice, stream);
}

extern "C" void copyGpuToCpuOnStream(int8_t *dst, const int8_t *src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(int8_t), cudaMemcpyDeviceToHost, stream);
}

extern "C" void copy2DTensorCpuToGpuOnStream(int8_t *dst, const int8_t *src, int rows, int cols, cudaStream_t stream) {
    size_t size = rows * cols * sizeof(int8_t);
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

extern "C" void copy2DTensorGpuToCpuOnStream(int8_t *dst, const int8_t *src, int rows, int cols, cudaStream_t stream) {
    size_t size = rows * cols * sizeof(int8_t);
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

extern "C" void copyCpuToGpuOnStream_float(float *dst, const float *src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyHostToDevice, stream);
}

extern "C" void copyGpuToCpuOnStream_float(float *dst, const float *src, int n, cudaStream_t stream) {
    cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
}

extern "C" void copy2DTensorCpuToGpuOnStream_float(float *dst, const float *src, int rows, int cols, cudaStream_t stream) {
    size_t size = rows * cols * sizeof(float);
    cudaMemcpyAsync(dst, src, size, cudaMemcpyHostToDevice, stream);
}

extern "C" void copy2DTensorGpuToCpuOnStream_float(float *dst, const float *src, int rows, int cols, cudaStream_t stream) {
    size_t size = rows * cols * sizeof(float);
    cudaMemcpyAsync(dst, src, size, cudaMemcpyDeviceToHost, stream);
}

extern "C" void synchronizeStream(cudaStream_t stream) {
    cudaStreamSynchronize(stream);
}

extern "C" void destroyStream(cudaStream_t stream) {
    cudaStreamDestroy(stream);
}

void float_operate(){
    float a[] = {1.1, 2.2, 3.3};
    float b[] = {4.4, 5.5, 6.6};
    float result[3];

    float *d_result; // Device pointer for the result array

    // Calculate the sum of the float arrays on the CPU
    addFloats(a, b, result, 3);

    // Allocate memory on the device
    cudaMalloc((void **)&d_result, sizeof(result));

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    copyCpuToGpuOnStream_float(d_result, result, 3, stream);

    // Output the results on the host
    printf("Result of adding two float arrays:\n");
    for (int i = 0; i < 3; i++) {
        printf("%f + %f = %f\n", a[i], b[i], result[i]);
    }

    copyGpuToCpuOnStream_float(result, d_result, 3, stream);
    // Cleanup
    cudaFree(d_result);
    cudaStreamDestroy(stream);
}

int main(){
    float_operate();
}
