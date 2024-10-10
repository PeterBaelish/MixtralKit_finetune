#include <iostream>
#include <vector>
#include <thread>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
using namespace std;

//g++ -fPIC -shared -o libtensorcompress.so tensor_compress.cpp 

extern "C"{
    /*
    void compressTensor(uint8_t* A, 
                    bool* mask, 
                    uint8_t* output, 
                    int width, int height, int sp_size,
                    int numThreads) {

        size_t y = width; //14336
        std::vector<std::thread> threads;
        std::vector<int> outputIndices(numThreads);

        // Pre-calculate the starting index in the output array for each thread
        int outputIndex = 0;
        for (int i = 0; i < y; ++i) {
            if (i % (y / numThreads) == 0) {
                outputIndices[i / (y / numThreads)] = outputIndex;
            }
            if (mask[i]) {
                outputIndex += height;
            }
        }

        auto worker = [&](int start, int end, int threadIndex) {
            int localOutputIndex = outputIndices[threadIndex];
            for (int i = start; i < end; ++i) {
                if (mask[i]) {
                    memcpy(output + localOutputIndex, A + i * height, height * sizeof(uint8_t));
                    localOutputIndex += height;
                }
            }
        };

        size_t blockSize = y / numThreads;
        for (int i = 0; i < numThreads; ++i) {
            int start = i * blockSize;
            int end = (i == numThreads - 1) ? y : (i + 1) * blockSize;

            threads.emplace_back(worker, start, end, i);
        }

        for (auto& t : threads) {
            t.join();
        }
    }*/
    // this is fastest
    void compressTensor(uint8_t* A, 
                    int* mask, 
                    uint8_t* output, 
                    int width, int height, int sp_size,
                    int numThreads) {

        std::vector<std::thread> threads;
        std::vector<int> outputIndices(numThreads);

        auto worker = [&](int start, int end, int threadIndex) {
            for (int i = start; i < end; ++i) {
                // cout << threadIndex << " " << i << " " << mask[i] << endl;
                memcpy(output + i * height, A + mask[i*2] * height, height * sizeof(uint8_t));
                /*
                for (int j = 0; j < width; j++) {
                    memcpy(output + j * sp_size + i, A + mask[i*2] + height*j, sizeof(uint8_t));
                }*/
            }
        };

        size_t blockSize = sp_size / numThreads;
        for (int i = 0; i < numThreads; ++i) {
            int start = i * blockSize;
            int end = (i == numThreads - 1) ? sp_size : (i + 1) * blockSize;

            threads.emplace_back(worker, start, end, i);
        }

        for (auto& t : threads) {
            t.join();
        }
    }
}


int main() {
    std::vector<std::vector<uint8_t>> A = {{1, 2}, {3, 4}, {5, 6}};
    std::vector<bool> mask = {true, false, true};
    std::vector<std::vector<uint8_t>> output;

    int numThreads = 2; // 可以自定义线程数量
    // compressTensor(A, mask, output, numThreads);

    // 打印结果
    for (const auto& row : output) {
        for (uint8_t elem : row) {
            std::cout << elem << ' ';
        }
        std::cout << std::endl;
    }

    return 0;
}
