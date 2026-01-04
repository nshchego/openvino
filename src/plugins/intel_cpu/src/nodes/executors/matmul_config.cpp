// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <vector>

#include "matmul_config.hpp"
#include "utils/serialization/serializers/any.hpp"
#include "utils/serialization/serializers/vector.hpp"

namespace ov::intel_cpu {

void MatMulAttrs::save(BinaryOutputBuffer& out_buf) const {
    out_buf.dump_position();  // TODO: remove

    out_buf << transposeA;
    out_buf << transposeB;
    out_buf << withBias;
    out_buf << weightsNonTransposed;
    out_buf << sparseWeights;
    out_buf << constantWeights;
    out_buf << fcSemantic;
    out_buf << dynamicQuantizationGroupSize;
    out_buf << dqScales;
    out_buf << postOps;

    out_buf.dump_position();  // TODO: remove
}

void MatMulAttrs::load(BinaryInputBuffer& in_buf) {
    in_buf.check_position();  // TODO: Remove

    in_buf >> transposeA;
    in_buf >> transposeB;
    in_buf >> withBias;
    in_buf >> weightsNonTransposed;
    in_buf >> sparseWeights;
    in_buf >> constantWeights;
    in_buf >> fcSemantic;
    in_buf >> dynamicQuantizationGroupSize;
    in_buf >> dqScales;
    in_buf >> postOps;

    in_buf.check_position();  // TODO: Remove
}

using MatMulConfig = executor::Config<MatMulAttrs>;
}  // namespace ov::intel_cpu
