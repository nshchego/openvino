// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "buffer.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

template <typename BufferType>
class Serializer<BufferType, ov::element::Type, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::element::Type& et) {
        auto type_t = (ov::element::Type_t)et;
        buffer.write(std::addressof(type_t), sizeof(ov::element::Type_t));
    }
};

template <typename BufferType>
class Serializer<BufferType, ov::element::Type, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::element::Type& et) {
        element::Type_t type_t = element::dynamic;
        buffer.read(std::addressof(type_t), sizeof(ov::element::Type_t));
        et = ov::element::Type(type_t);
    }
};


template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value && std::is_enum_v<T>>::type> {
public:
    static void save(BufferType& buffer, const T& enm) {
        buffer.write(std::addressof(enm), sizeof(enm));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value && std::is_enum_v<T>>::type> {
public:
    static void load(BufferType& buffer, T& enm) {
        buffer.read(std::addressof(enm), sizeof(enm));
    }
};

}  // namespace ov::intel_cpu
