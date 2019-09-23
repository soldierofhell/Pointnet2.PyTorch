#ifndef _SAMPLING_GPU_H
#define _SAMPLING_GPU_H

#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include<vector>


void furthest_point_sampling_kernel_launcher(int_64 b, int_64 n, int_64 m, 
    const double *dataset, double *temp, int_64 *idxs, cudaStream_t stream);

#endif
