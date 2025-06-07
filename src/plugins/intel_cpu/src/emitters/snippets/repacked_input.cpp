// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "repacked_input.hpp"

#include <memory>
#include <utility>

#include "cpu_types.h"
#include "memory_desc/cpu_blocked_memory_desc.h"
#include "openvino/core/except.hpp"
#include "utils/serialization/vector_serializer.hpp"

namespace ov::intel_cpu {

RepackedInput::RepackedInput(std::shared_ptr<const RepackedInputKernel> kernel,
                             CpuBlockedMemoryDescPtr desc,
                             VectorDims in_offsets,
                             VectorDims out_offsets)
    : m_kernel(std::move(kernel)),
      m_desc(std::move(desc)),
      m_in_offsets(std::move(in_offsets)),
      m_out_offsets(std::move(out_offsets)) {
    OPENVINO_ASSERT(m_in_offsets.size() == m_out_offsets.size(), "Incorrect size of offsets");
    OPENVINO_ASSERT(m_desc, "Descriptor is empty");
}

const CpuBlockedMemoryDescPtr& RepackedInput::desc() const {
    return m_desc;
}

const VectorDims& RepackedInput::in_offsets() const {
    return m_in_offsets;
}

const VectorDims& RepackedInput::out_offsets() const {
    return m_out_offsets;
}

void RepackedInput::save(BinaryOutputBuffer& ob) const {
    ob << m_in_offsets;
    ob << m_out_offsets;
    // ob << m_desc;
    // ob << m_kernel;
}

void RepackedInput::load(BinaryInputBuffer& ib) {
    ib >> m_in_offsets;
    ib >> m_out_offsets;
    // ib >> m_desc;
    // ib >> m_kernel;
}

}  // namespace ov::intel_cpu
