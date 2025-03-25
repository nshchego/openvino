// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "utils/serialization/internal_types.hpp"

namespace ov::intel_cpu {

void MemoryDesc::save(BinaryOutputBuffer& ob) const {
    ob << type;
    ob << shape;
    ob << status;
}

void MemoryDesc::load(BinaryInputBuffer& ib) {
    ib >> type;
    ib >> shape;
    ib >> status;
}

}  // namespace ov::intel_cpu
