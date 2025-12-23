// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node_config.h"
#include "utils/serialization/polymorphic_serializer.hpp"
#include "utils/serialization/string_serializer.hpp"
#include "utils/serialization/vector_serializer.hpp"

namespace ov::intel_cpu {

void PortDescGeneric::save(BinaryOutputBuffer& out_buf) const {
    out_buf << m_mem_desc;
    //     out_buf.dump_position();  // TODO: remove
}

void PortDescGeneric::load(BinaryInputBuffer& in_buf) {
    in_buf >> m_mem_desc;
    // in_buf.check_position();  // TODO: Remove
}

void PortDescBlocked::save(BinaryOutputBuffer& out_buf) const {
    out_buf << m_mem_desc;
        out_buf.dump_position();  // TODO: remove
    uint32_t tmp = m_cmp_mask.to_ulong();
    out_buf << tmp;
        out_buf.dump_position();  // TODO: remove
}

void PortDescBlocked::load(BinaryInputBuffer& in_buf) {
    in_buf >> m_mem_desc;
    in_buf.check_position();  // TODO: Remove
    uint32_t tmp = 0U;
    in_buf >> tmp;
    m_cmp_mask = CmpMask(tmp);
    in_buf.check_position();  // TODO: Remove
}

void PortConfig::save(BinaryOutputBuffer& out_buf) const {
    out_buf << m_in_place_port;
        out_buf.dump_position();  // TODO: remove
    out_buf << m_constant;
        out_buf.dump_position();  // TODO: remove
    out_buf << m_port_desc;
        out_buf.dump_position();  // TODO: remove
}

void PortConfig::load(BinaryInputBuffer& in_buf) {
    in_buf >> m_in_place_port;
    in_buf.check_position();  // TODO: Remove
    in_buf >> m_constant;
    in_buf.check_position();  // TODO: Remove
    in_buf >> m_port_desc;
    in_buf.check_position();  // TODO: Remove
}

void NodeConfig::save(BinaryOutputBuffer& out_buf) const {
    out_buf << inConfs;
        out_buf.dump_position();  // TODO: remove
    out_buf << outConfs;
        out_buf.dump_position();  // TODO: remove
}

void NodeConfig::load(BinaryInputBuffer& in_buf) {
    in_buf >> inConfs;
    in_buf.check_position();  // TODO: Remove
    in_buf >> outConfs;
    in_buf.check_position();  // TODO: Remove
}

}  // namespace ov::intel_cpu

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::PortDescGeneric)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::PortDescBlocked)
