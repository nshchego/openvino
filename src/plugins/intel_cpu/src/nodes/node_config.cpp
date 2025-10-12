// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "node_config.h"
#include "utils/serialization/polymorphic_serializer.hpp"
#include "utils/serialization/string_serializer.hpp"
#include "utils/serialization/vector_serializer.hpp"

namespace ov::intel_cpu {

void PortDescGeneric::save(BinaryOutputBuffer& ob) const {
    ob << m_mem_desc;
    // ob << ob.get_pos();  // TODO: remove
}

void PortDescGeneric::load(BinaryInputBuffer& in_buf) {
    in_buf >> m_mem_desc;
    // validate_stream_offset(in_buf);  // TODO: Remove
}

void PortDescBlocked::save(BinaryOutputBuffer& ob) const {
    ob << m_mem_desc;
    ob << ob.get_pos();  // TODO: remove
    uint32_t tmp = m_cmp_mask.to_ulong();
    ob << tmp;
    ob << ob.get_pos();  // TODO: remove
}

void PortDescBlocked::load(BinaryInputBuffer& in_buf) {
    in_buf >> m_mem_desc;
    validate_stream_offset(in_buf);  // TODO: Remove
    uint32_t tmp = 0U;
    in_buf >> tmp;
    m_cmp_mask = CmpMask(tmp);
    validate_stream_offset(in_buf);  // TODO: Remove
}

void PortConfig::save(BinaryOutputBuffer& ob) const {
    ob << m_in_place_port;
    ob << ob.get_pos();  // TODO: remove
    ob << m_constant;
    ob << ob.get_pos();  // TODO: remove
    ob << m_port_desc;
    ob << ob.get_pos();  // TODO: remove
}

void PortConfig::load(BinaryInputBuffer& in_buf) {
    in_buf >> m_in_place_port;
    validate_stream_offset(in_buf);  // TODO: Remove
    in_buf >> m_constant;
    validate_stream_offset(in_buf);  // TODO: Remove
    in_buf >> m_port_desc;
    validate_stream_offset(in_buf);  // TODO: Remove
}

void NodeConfig::save(BinaryOutputBuffer& ob) const {
    ob << inConfs;
    ob << ob.get_pos();  // TODO: remove
    ob << outConfs;
    ob << ob.get_pos();  // TODO: remove
}

void NodeConfig::load(BinaryInputBuffer& in_buf) {
    in_buf >> inConfs;
    validate_stream_offset(in_buf);  // TODO: Remove
    in_buf >> outConfs;
    validate_stream_offset(in_buf);  // TODO: Remove
}

}  // namespace ov::intel_cpu

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::PortDescGeneric)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::PortDescBlocked)
