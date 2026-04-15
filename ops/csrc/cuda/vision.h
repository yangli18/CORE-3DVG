#pragma once
#include <torch/extension.h>

at::Tensor ml_nms_cuda(const at::Tensor boxes, float nms_overlap_thresh);