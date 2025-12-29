// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling.hpp"

#include <utility>

#include "nodes/executors/executor.hpp"
#include "utils/serialization/serializers/internal_types.hpp"
#include "utils/serialization/serializers/vector.hpp"

namespace ov::intel_cpu {

PoolingExecutor::PoolingExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

void PoolingAttrs::save(BinaryOutputBuffer& out_buf) const {
    out_buf << exclude_pad;
    out_buf << auto_pad;
    out_buf << pad_type;
    out_buf << algorithm;
    out_buf << rounding;
    out_buf << stride;
    out_buf << kernel;
    out_buf << dilation;
    out_buf << data_pad_begin;
    out_buf << data_pad_end;
    out_buf << effective_pad_begin;
    out_buf << effective_pad_end;
    out_buf << effective_dilation;
}

void PoolingAttrs::load(BinaryInputBuffer& in_buf) {
    in_buf >> exclude_pad;
    in_buf >> auto_pad;
    in_buf >> pad_type;
    in_buf >> algorithm;
    in_buf >> rounding;
    in_buf >> stride;
    in_buf >> kernel;
    in_buf >> dilation;
    in_buf >> data_pad_begin;
    in_buf >> data_pad_end;
    in_buf >> effective_pad_begin;
    in_buf >> effective_pad_end;
    in_buf >> effective_dilation;
}

}  // namespace ov::intel_cpu
