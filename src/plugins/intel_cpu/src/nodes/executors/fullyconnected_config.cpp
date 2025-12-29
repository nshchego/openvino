// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "fullyconnected_config.hpp"
#include "utils/serialization/internal_types.hpp"

namespace ov::intel_cpu {

void FCAttrs::save(BinaryOutputBuffer& out_buf) const {
    out_buf.dump_position();  // TODO: remove

    out_buf << withBias;
    out_buf << weightsNonTransposed;
    out_buf << sparseWeights;
    out_buf << dynamicQuantizationGroupSize;
    out_buf << constantWeights;
    out_buf << modelType;
    // out_buf << postOps;

    out_buf.dump_position();  // TODO: remove
}

void FCAttrs::load(BinaryInputBuffer& in_buf) {
    in_buf.check_position();  // TODO: remove

    in_buf >> withBias;
    in_buf >> weightsNonTransposed;
    in_buf >> sparseWeights;
    in_buf >> dynamicQuantizationGroupSize;
    in_buf >> constantWeights;
    in_buf >> modelType;
    // in_buf >> postOps;

    in_buf.check_position();  // TODO: remove
}

}  // namespace ov::intel_cpu
