cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(pointnet2_ts)

find_package(Torch REQUIRED)

# Define our library target
add_library(furthest_point_sampling SHARED sampling_ts.cpp)
# Enable C++11
target_compile_features(furthest_point_sampling PRIVATE cxx_range_for)
# Link against LibTorch
target_link_libraries(furthest_point_sampling "${TORCH_LIBRARIES}")