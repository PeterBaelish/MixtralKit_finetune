#include "add_floats.h"

void addFloats(const float *a, const float *b, float *result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}