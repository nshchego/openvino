// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffers.hpp"

#include "memory_desc/cpu_blocked_memory_desc.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"
#include "memory_desc/empty_memory_desc.h"

#include "shape_inference/custom/adaptive_pooling.hpp"
#include "shape_inference/custom/color_convert.hpp"
#include "shape_inference/custom/convolution.hpp"
#include "shape_inference/custom/eltwise.hpp"
#include "shape_inference/custom/fullyconnected.hpp"
#include "shape_inference/custom/gather.hpp"
#include "shape_inference/custom/matmul.hpp"
#include "shape_inference/custom/ngram.hpp"
#include "shape_inference/custom/one_hot.hpp"
#include "shape_inference/custom/priorbox_clustered.hpp"
#include "shape_inference/custom/priorbox.hpp"
#include "shape_inference/custom/reshape.hpp"
#include "shape_inference/custom/shapeof.hpp"
#include "shape_inference/custom/strided_slice.hpp"
#include "shape_inference/custom/transpose.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"


BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::CpuBlockedMemoryDesc)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::DnnlBlockedMemoryDesc)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::EmptyMemoryDesc)

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::ShapeInferPassThrough)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::AdaptivePoolingShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::ColorConvertShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::ConvolutionShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::EltwiseShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::FCShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::GatherShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::MMShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::NgramShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::OneHotShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::PriorBoxClusteredShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::PriorBoxShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::ReshapeShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::ShapeOfShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::SqueezeShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::StridedSliceShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::TransposeShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::UnsqueezeShapeInfer)
