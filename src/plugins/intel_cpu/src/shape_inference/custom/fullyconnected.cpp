// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fullyconnected.hpp"

#include <cstddef>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "shape_inference/shape_inference_status.hpp"

namespace ov::intel_cpu::node {

Result FCShapeInfer::infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                           [[maybe_unused]] const std::unordered_map<size_t, MemoryPtr>& data_dependency) {
    const VectorDims& activationShape = input_shapes[0].get();
    const VectorDims& weightShape = input_shapes[1].get();
    size_t activationRank = activationShape.size();
    size_t channelRank = 1;

    // activation   weight    output_shape
    // NCHW         CoCHW     NCo
    // TNC          CoC       TNCo
    // NC           CoC       NCo
    VectorDims outputShape(out_rank, 1);
    // set Co
    outputShape.back() = std::accumulate(weightShape.begin(), weightShape.end() - 1, 1, std::multiplies<>());
    // set batch dims
    size_t batchRank = activationRank - channelRank;
    size_t startIdx = out_rank - batchRank - 1;
    for (size_t i = 0; i < batchRank; i++) {
        outputShape[i + startIdx] = activationShape[i];
    }

    return {{std::move(outputShape)}, ShapeInferStatus::success};
}

void FCShapeInfer::save(BinaryOutputBuffer& out_buf) const {
    out_buf << out_rank;
}

void FCShapeInfer::load(BinaryInputBuffer& in_buf) {
    in_buf >> out_rank;
}

}  // namespace ov::intel_cpu::node

// auto& loader = LoaderStorage<BinaryInputBuffer, std::function<void(BinaryInputBuffer&, std::unique_ptr<void, VoidDeleter<void>>&)>>::instance();
// auto load_fn = [](BinaryInputBuffer& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr) {
//     auto derived_ptr = std::unique_ptr<FCShapeInfer>(new FCShapeInfer(buffer));
//     dst_ptr.reset(derived_ptr.release());
// }
// loader.set_load_function({FCShapeInfer::get_type_info_s(), load_fn});
