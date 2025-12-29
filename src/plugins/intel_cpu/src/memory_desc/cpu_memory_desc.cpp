// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "utils/serialization/serializers/internal_types.hpp"

namespace ov::intel_cpu {

void MemoryDesc::save(BinaryOutputBuffer& out_buf) const {
    out_buf.dump_position();  // TODO: remove

    out_buf << type;
    out_buf << shape;
    out_buf << status;

    out_buf.dump_position();  // TODO: remove
}

void MemoryDesc::load(BinaryInputBuffer& in_buf) {
    in_buf.check_position();  // TODO: remove

    in_buf >> type;
    in_buf >> shape;
    in_buf >> status;

    in_buf.check_position();  // TODO: remove
}

}  // namespace ov::intel_cpu
