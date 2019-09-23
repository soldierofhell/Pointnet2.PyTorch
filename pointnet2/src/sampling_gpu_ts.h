#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include<vector>


void furthest_point_sampling_kernel_launcher(int64_t b, int64_t n, int64_t m, 
    const double *dataset, double *temp, int64_t *idxs, cudaStream_t stream);

#endif
