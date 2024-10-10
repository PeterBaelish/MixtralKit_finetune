import os

if os.path.exists("/workspace/MixtralKit_finetune/output_data.txt"):

    integers = []
    with open("/workspace/MixtralKit_finetune/output_data.txt", "r") as file:
        for line in file:
            integers.append(float(line.strip()))

    cache_predict_hit = [0]*32
    n_token = len(integers)//32

    for i in range(n_token):
        for j in range(32):
            cache_predict_hit[j] += integers[i*32+j]

    for i in range(32):
        cache_predict_hit[i] /= n_token
        cache_predict_hit[i] /= 2

    print(cache_predict_hit)

    os.remove("/workspace/MixtralKit_finetune/output_data.txt")
