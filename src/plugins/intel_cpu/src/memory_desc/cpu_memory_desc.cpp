// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cpu_memory_desc.h"
#include "utils/serialization/internal_types.hpp"

namespace ov::intel_cpu {

void MemoryDesc::save(BinaryOutputBuffer& ob) const {
    ob << ob.get_pos();  // TODO: remove

    ob << type;
    ob << ob.get_pos();  // TODO: remove
    ob << shape;
    ob << ob.get_pos();  // TODO: remove
    ob << status;

    ob << ob.get_pos();  // TODO: remove
}

void MemoryDesc::load(BinaryInputBuffer& ib) {
    validate_stream_offset(ib);  // TODO: remove

    ib >> type;
    validate_stream_offset(ib);  // TODO: remove
    ib >> shape;
    validate_stream_offset(ib);  // TODO: remove
    ib >> status;

    validate_stream_offset(ib);  // TODO: remove
}

}  // namespace ov::intel_cpu
