// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "convolution_config.hpp"
#include "utils/serialization/serializers/any.hpp"
#include "utils/serialization/serializers/internal_types.hpp"
#include "utils/serialization/serializers/vector.hpp"

namespace ov::intel_cpu {

void ConvAttrs::save(BinaryOutputBuffer& out_buf) const {
    out_buf << stride;
    out_buf << dilation;
    out_buf << paddingL;
    out_buf << paddingR;
    out_buf << autoPadding;
    out_buf << withBias;
    out_buf << weightsNonTransposed;
    out_buf << isGrouped;
    out_buf << isGraphQuantized;
    out_buf << fcSemantic;
    out_buf << constantWeights;
    out_buf << inputZeroPointsType;
    out_buf << dqScales;
    out_buf << postOps;
}

void ConvAttrs::load(BinaryInputBuffer& in_buf) {
    in_buf >> stride;
    in_buf >> dilation;
    in_buf >> paddingL;
    in_buf >> paddingR;
    in_buf >> autoPadding;
    in_buf >> withBias;
    in_buf >> weightsNonTransposed;
    in_buf >> isGrouped;
    in_buf >> isGraphQuantized;
    in_buf >> fcSemantic;
    in_buf >> constantWeights;
    in_buf >> inputZeroPointsType;
    in_buf >> dqScales;
    in_buf >> postOps;
}

}  // namespace ov::intel_cpu
