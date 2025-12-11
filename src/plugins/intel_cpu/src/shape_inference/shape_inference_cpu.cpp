// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shape_inference/shape_inference_cpu.hpp"

#include <memory>
#include <utility>

#include "openvino/core/coordinate_diff.hpp"
#include "openvino/core/node.hpp"
#include "shape_inference/shape_inference.hpp"
#include "utils/serialization/bind.hpp"

namespace ov::intel_cpu {
NgraphShapeInferFactory::NgraphShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}

ShapeInferPtr NgraphShapeInferFactory::makeShapeInfer() const {
    return make_shape_inference(m_op);
}

const ov::CoordinateDiff ShapeInferEmptyPads::m_emptyVec = {};

// auto& loader = LoaderStorage<BinaryInputBuffer, std::function<void(BinaryInputBuffer&, std::unique_ptr<void, VoidDeleter<void>>&)>>::instance();
// auto load_fn = [](BinaryInputBuffer& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr) {
//     auto derived_ptr = std::unique_ptr<ShapeInferEmptyPads>(new ShapeInferEmptyPads());
//     dst_ptr.reset(derived_ptr.release());
// }
// loader.set_load_function({ShapeInferEmptyPads::get_type_info_s(), load_fn});
}  // namespace ov::intel_cpu

// BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::ShapeInferEmptyPads)
