// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0mvn
//

#include "eltwise.hpp"

#include "utils/serialization/internal_types.hpp"
#include <utility>

namespace ov::intel_cpu {

EltwiseExecutor::EltwiseExecutor(ExecutorContext::CPtr context) : context(std::move(context)) {}

void EltwiseAttrs::save(BinaryOutputBuffer& ob) const {
    ob << algorithm;
    ob << alpha;
    ob << beta;
    ob << gamma;
}

void EltwiseAttrs::load(BinaryInputBuffer& ib) {
    ib >> algorithm;
    ib >> alpha;
    ib >> beta;
    ib >> gamma;
}

}  // namespace ov::intel_cpu
