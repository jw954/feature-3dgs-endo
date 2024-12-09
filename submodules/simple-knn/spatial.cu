/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

// newest version of simple knn distCUDA, find solution on https://github.com/camenduru/simple-knn/blob/main/spatial.cu

// Solved a problem that the distCUDA2 not work on CUDA devices other than "cuda:0".

#include "spatial.h"
#include "simple_knn.h"

#include <cuda_runtime.h>  // Include the CUDA runtime header for cudaSetDevice()

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  // Determine which device the tensor is on
  auto device = points.device();
  int device_index = device.index(); // Get the index of the device

  // Set the current CUDA device to the device where 'points' is located
  cudaSetDevice(device_index);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}