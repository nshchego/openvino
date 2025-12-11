// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffers.hpp"
#include "shape_inference/custom/fullyconnected.hpp"
#include "shape_inference/custom/reshape.hpp"
#include "shape_inference/custom/transpose.hpp"
#include "shape_inference/shape_inference_pass_through.hpp"

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::ShapeInferPassThrough)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::FCShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::ReshapeShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::SqueezeShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::UnsqueezeShapeInfer)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::node::TransposeShapeInfer)
