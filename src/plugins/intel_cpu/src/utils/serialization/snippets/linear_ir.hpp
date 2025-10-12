// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "../buffers.hpp"
#include "openvino/core/type/element_type.hpp"
#include "snippets/lowered/linear_ir.hpp"

namespace ov::intel_cpu {

template <typename BufferType>
class Serializer<BufferType, snippets::lowered::Config, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
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
class Serializer<BufferType, snippets::lowered::Config, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
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

// template <typename BufferType, typename T>
// class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value && std::is_enum_v<T>>::type> {
// public:
//     static void save(BufferType& buffer, const T& enm) {
// // printf("-WRITE enum-\n");
//         buffer.write(std::addressof(enm), sizeof(enm));
//     }
// };

// template <typename BufferType, typename T>
// class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value && std::is_enum_v<T>>::type> {
// public:
//     static void load(BufferType& buffer, T& enm) {
// // printf("-READ enum-\n");
//         buffer.read(std::addressof(enm), sizeof(enm));
//     }
// };

}  // namespace ov::intel_cpu
