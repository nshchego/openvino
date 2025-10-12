// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert.hpp"

#include <utility>

#include "nodes/executors/executor.hpp"
#include "utils/serialization/internal_types.hpp"

namespace ov::intel_cpu {

ConvertExecutor::ConvertExecutor(ov::intel_cpu::ExecutorContext::CPtr context)
    : convertContext(std::move(context)) {}

void ConvertParams::save(BinaryOutputBuffer& ob) const {
ob << ob.get_pos();  // TODO: remove

    ob << srcPrc;
    ob << origPrc;
    ob << dstPrc;
    ob << size;

ob << ob.get_pos();  // TODO: remove
}

void ConvertParams::load(BinaryInputBuffer& in_buf) {
validate_stream_offset(in_buf);  // TODO: remove

    in_buf >> srcPrc;
    in_buf >> origPrc;
    in_buf >> dstPrc;
    in_buf >> size;

validate_stream_offset(in_buf);  // TODO: remove
}

}  // namespace ov::intel_cpu
