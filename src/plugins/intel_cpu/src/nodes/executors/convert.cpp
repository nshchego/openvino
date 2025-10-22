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
ob.dump_position();  // TODO: remove

    ob << srcPrc;
    ob << origPrc;
    ob << dstPrc;
    ob << size;

ob.dump_position();  // TODO: remove
}

void ConvertParams::load(BinaryInputBuffer& in_buf) {
in_buf.check_position();  // TODO: remove

    in_buf >> srcPrc;
    in_buf >> origPrc;
    in_buf >> dstPrc;
    in_buf >> size;

in_buf.check_position();  // TODO: remove
}

}  // namespace ov::intel_cpu
