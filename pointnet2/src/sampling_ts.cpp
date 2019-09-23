#include <torch/serialize/tensor.h>
#include <ATen/cuda/CUDAContext.h>
#include <vector>
#include <THC/THC.h>
#include <torch/script.h>

#include "sampling_gpu_ts.h"


extern THCState *state;

static auto registry =
  torch::RegisterOperators("pointnet2_ts::furthest_point_sampling_wrapper", &furthest_point_sampling_wrapper);


int furthest_point_sampling_wrapper(int64_t b, int64_t n, int64_t m, 
    at::Tensor points_tensor, at::Tensor temp_tensor, at::Tensor idx_tensor) {

    const double *points = points_tensor.data<double>();
    double *temp = temp_tensor.data<double>();
    int64_t *idx = idx_tensor.data<int64_t>();

    cudaStream_t stream = THCState_getCurrentStream(state);
    furthest_point_sampling_kernel_launcher(b, n, m, points, temp, idx, stream);
    return 1;
}
