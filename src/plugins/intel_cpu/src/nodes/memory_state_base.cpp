// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_state_base.h"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/variable_extension.hpp"
#include "utils/serialization/serializers/string.hpp"

using namespace ov::intel_cpu::node;

MemoryNode::MemoryNode(const std::shared_ptr<ov::Node>& op) {
    if (auto assignOp = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
        m_id = assignOp->get_variable_id();
    } else {
        OPENVINO_THROW("Unexpected ov::Node type: ", op->get_type_info().name, " in MemoryNode");
    }
}

MemoryNode::MemoryNode(BinaryInputBuffer& in_buf) {
    load(in_buf);
}

void MemoryNode::save(BinaryOutputBuffer& out_buf) const {
    out_buf.dump_position();  // TODO: remove

    out_buf << m_id;

    out_buf.dump_position();  // TODO: remove
}

void MemoryNode::load(BinaryInputBuffer& in_buf) {
    in_buf.check_position();  // TODO: remove

    in_buf >> m_id;

    in_buf.check_position();  // TODO: remove
}
