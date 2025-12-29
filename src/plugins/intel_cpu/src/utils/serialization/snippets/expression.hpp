// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffers.hpp"
#include "snippets\lowered\expression.hpp"

namespace ov::intel_cpu {

class ExpressionSerializer : public snippets::lowered::Expression::IExpressionSerializer {
    ExpressionSerializer() {

    }
};

template <typename BufferType>
class Serializer<BufferType, ExpressionSerializer, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const snippets::lowered::Config& conf) {
        buffer << conf.m_need_fill_tail_register;
        buffer << conf.m_enable_domain_optimization;
        buffer << conf.m_min_parallel_work_amount;
        buffer << conf.m_min_kernel_work_amount;
        buffer << conf.m_are_buffers_optimized;
        buffer << conf.m_manual_build_support;
    }
};

template <typename BufferType>
class Serializer<BufferType, ExpressionSerializer, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, snippets::lowered::Config& conf) {
        buffer >> conf.m_need_fill_tail_register;
        buffer >> conf.m_enable_domain_optimization;
        buffer >> conf.m_min_parallel_work_amount;
        buffer >> conf.m_min_kernel_work_amount;
        buffer >> conf.m_are_buffers_optimized;
        buffer >> conf.m_manual_build_support;
    }
};

}  // namespace ov::intel_cpu
