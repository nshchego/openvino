// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "executor.hpp"

#include <string>

namespace ov::intel_cpu {

std::string ExecutorTypeToString(const ExecutorType type) {
#define CASE(_type)           \
    case ExecutorType::_type: \
        return #_type;
    switch (type) {
        CASE(Undefined);
        CASE(Graph);
        CASE(Reference);
        CASE(Common);
        CASE(Jit);
        CASE(Dnnl);
        CASE(Acl);
        CASE(Mlas);
        CASE(Shl);
        CASE(Kleidiai);
    }
#undef CASE
    return "Undefined";
}

ExecutorType ExecutorTypeFromString(const std::string& typeStr) {
#define CASE(_type)                 \
    if (typeStr == #_type) {        \
        return ExecutorType::_type; \
    }
    CASE(Undefined);
    CASE(Graph);
    CASE(Reference);
    CASE(Common);
    CASE(Jit);
    CASE(Dnnl);
    CASE(Acl);
    CASE(Mlas);
    CASE(Shl);
    CASE(Kleidiai);
#undef CASE
    return ExecutorType::Undefined;
}


void ExecutorContext::save(BinaryOutputBuffer& out_buf) const {
    out_buf.dump_position();  // TODO: remove
    out_buf << m_impl_priorities;
    out_buf.dump_position();  // TODO: remove
    out_buf << m_num_numa_nodes;
    out_buf.dump_position();  // TODO: remove
    out_buf << m_cur_numa_node_id;
    // out_buf << m_private_weight_cache;
    out_buf.dump_position();  // TODO: remove
}

void ExecutorContext::load(BinaryInputBuffer& in_buf) {
    in_buf.check_position();  // TODO: Remove
    in_buf >> m_impl_priorities;
    in_buf.check_position();  // TODO: Remove
    in_buf >> m_num_numa_nodes;
    in_buf.check_position();  // TODO: Remove
    in_buf >> m_cur_numa_node_id;
    // in_buf >> m_private_weight_cache;
    in_buf.check_position();  // TODO: Remove
}

}  // namespace ov::intel_cpu

BIND_BINARY_BUFFER_WITH_TYPE(ExecutorContext)
