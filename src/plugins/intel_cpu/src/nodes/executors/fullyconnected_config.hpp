// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>

#include "config.h"
#include "executor_config.hpp"
#include "post_ops.hpp"

namespace ov::intel_cpu {

// @todo require explicit initialization of all the attributes?
struct FCAttrs {
    // @todo probably we don't want with bias flag, since this information is already
    // a part of src memory descs
    bool withBias = false;
    bool weightsNonTransposed = false;
    bool sparseWeights = false;
    uint64_t dynamicQuantizationGroupSize = 0;
    bool constantWeights = true;

    ov::intel_cpu::Config::ModelType modelType = ov::intel_cpu::Config::ModelType::Unknown;

    PostOps postOps;

    void save(BinaryOutputBuffer& out_buf) const {
out_buf.dump_position();  // TODO: remove

        // out_buf << withBias;
        // out_buf << weightsNonTransposed;
        // out_buf << sparseWeights;
        // out_buf << dynamicQuantizationGroupSize;
        // out_buf << constantWeights;
        // out_buf << modelType;
        // out_buf << postOps;

out_buf.dump_position();  // TODO: remove
    }

    void load(BinaryInputBuffer& in_buf) {
in_buf.check_position();  // TODO: remove

        // in_buf >> withBias;
        // in_buf >> weightsNonTransposed;
        // in_buf >> sparseWeights;
        // in_buf >> dynamicQuantizationGroupSize;
        // in_buf >> constantWeights;
        // in_buf >> modelType;
        // in_buf >> postOps;

in_buf.check_position();  // TODO: remove
    }
};

using FCConfig = executor::Config<FCAttrs>;
}  // namespace ov::intel_cpu
