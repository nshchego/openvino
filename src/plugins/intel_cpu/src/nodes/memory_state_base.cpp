// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "memory_state_base.h"

#include <memory>

#include "openvino/core/except.hpp"
#include "openvino/core/node.hpp"
#include "openvino/op/util/variable_extension.hpp"

using namespace ov::intel_cpu::node;

MemoryNode::MemoryNode(const std::shared_ptr<ov::Node>& op) {
    if (auto assignOp = std::dynamic_pointer_cast<ov::op::util::VariableExtension>(op)) {
        m_id = assignOp->get_variable_id();
    } else {
        OPENVINO_THROW("Unexpected ov::Node type: ", op->get_type_info().name, " in MemoryNode");
    }
}

MemoryNode::MemoryNode(BinaryInputBuffer& in_buf) {
    // load(in_buf);
}

// void MemoryNode::save(BinaryOutputBuffer& ob) const {
//     Node::save(ob);

// ob << ob.get_pos();  // TODO: remove


// ob << ob.get_pos();  // TODO: remove
// }

// void MemoryNode::load(BinaryInputBuffer& in_buf) {
// validate_stream_offset(in_buf);  // TODO: remove

// validate_stream_offset(in_buf);  // TODO: remove
// }