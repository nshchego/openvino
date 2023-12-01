// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once
#include <vector>
#include "cpu_memory.h"
#include "openvino/core/shape.hpp"
#include "tensor_data_accessor.hpp"

namespace ov {
namespace intel_cpu {
/**
 * @brief cpu memory accessor implementing ov::ITensorAccessor to get data as tensor from cpu container.
 */
class MemoryAccessor : public ov::ITensorAccessor {
    using container_type = std::unordered_map<size_t, MemoryPtr>;

public:
    MemoryAccessor(const container_type& ptrs, const std::vector<int64_t>& ranks)
        : m_ptrs{ptrs}, m_ranks(ranks) {
if (m_ptrs.size() > 0) {
    printf("[CPU] MemoryAccessor ctr m_ptrs.size: %lu\n", m_ptrs.size());
    for (auto& ptr : m_ptrs) {
        if (ptr.second->getSize() == 7260269407829122528) {
            printf("ptr.second->getSize() == 7260269407829122528\n");
        }
        printf("    port: %lu; size: %lu; ptr: %p\n", ptr.first, ptr.second->getSize(), ptr.second->getData());
    }
}
        }

    ~MemoryAccessor() = default;

    ov::Tensor operator()(size_t port) const override {
        const auto t_iter = m_ptrs.find(port);
        if (t_iter != m_ptrs.cend()) {
            auto memPtr = t_iter->second;
            // use scalar shape {} instead of {1} if required by shapeInference
            const auto shape = (m_ranks[port] != 0) ? ov::Shape(memPtr->getStaticDims()) : ov::Shape();
if (memPtr->getDesc().getPrecision() == ov::element::string) {
    printf("[CPU][STRING] MemoryAccessor size: %lu; ptr: %p\n", memPtr->getSize(), memPtr->getData());
    auto strdata = reinterpret_cast<std::string *>(memPtr->getData());
    // for (size_t i = 0lu; i < memPtr->getSize() / 32; i++) {
    for (size_t i = 0lu; i < memPtr->getShape().getElementsCount(); i++) {
        std::cout << "    \"" << strdata[i] << "\"" << std::endl;
    }
}
            return {memPtr->getDesc().getPrecision(),
                    shape,
                    memPtr->getData()
                   };
        } else {
            return {};
        }
    }

private:
    const container_type& m_ptrs;              //!< Pointer to cpu memory pointers with op data.
    const std::vector<int64_t>& m_ranks;
};
}  // namespace intel_cpu
}  // namespace ov

