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
}

void PortDescGeneric::load(BinaryInputBuffer& ib) {
    ib >> m_mem_desc;
}

void PortDescBlocked::save(BinaryOutputBuffer& ob) const {
    ob << m_mem_desc;
    uint32_t tmp = m_cmp_mask.to_ulong();
    ob << tmp;
}

void PortDescBlocked::load(BinaryInputBuffer& ib) {
    ib >> m_mem_desc;
    uint32_t tmp = 0U;
    ib >> tmp;
    m_cmp_mask = CmpMask(tmp);
}

void PortConfig::save(BinaryOutputBuffer& ob) const {
    ob << m_in_place_port;
    ob << m_constant;
    ob << m_port_desc;
}

void PortConfig::load(BinaryInputBuffer& ib) {
    ib >> m_in_place_port;
    ib >> m_constant;
    ib >> m_port_desc;
}

void NodeConfig::save(BinaryOutputBuffer& ob) const {
    ob << inConfs;
    ob << outConfs;
}

void NodeConfig::load(BinaryInputBuffer& ib) {
    ib >> inConfs;
    ib >> outConfs;
}

}  // namespace ov::intel_cpu

BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::PortDescGeneric)
BIND_BINARY_BUFFER_WITH_TYPE(ov::intel_cpu::PortDescBlocked)
