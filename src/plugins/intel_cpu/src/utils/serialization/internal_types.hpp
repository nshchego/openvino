// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <bitset>

#include "buffers.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu {

template <typename BufferType>
class Serializer<BufferType, element::Type, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const ov::element::Type& et) {
// printf("-WRITE ov::element::Type-\n");
        auto type_t = (ov::element::Type_t)et;
        buffer.write(std::addressof(type_t), sizeof(ov::element::Type_t));
    }
};

template <typename BufferType>
class Serializer<BufferType, element::Type, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, ov::element::Type& et) {
// printf("-READ ov::element::Type-\n");
        element::Type_t type_t = element::dynamic;
        buffer.read(std::addressof(type_t), sizeof(ov::element::Type_t));
        et = ov::element::Type(type_t);
    }
};


template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value && std::is_enum_v<T>>::type> {
public:
    static void save(BufferType& buffer, const T& enm) {
// printf("-WRITE enum-\n");  // TODO: remove
        buffer.write(std::addressof(enm), sizeof(enm));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value && std::is_enum_v<T>>::type> {
public:
    static void load(BufferType& buffer, T& enm) {
// printf("-READ enum-\n");  // TODO: remove
        buffer.read(std::addressof(enm), sizeof(enm));
    }
};


template <typename BufferType, size_t Size>
class Serializer<BufferType, std::bitset<Size>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::bitset<Size>& bit_set) {
// printf("-WRITE bitset-\n");  // TODO: remove
        uint64_t val = bit_set.to_ullong();
        buffer << val;
    }
};

template <typename BufferType, size_t Size>
class Serializer<BufferType, std::bitset<Size>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::bitset<Size>& bit_set) {
// printf("-READ bitset-\n");  // TODO: remove
        uint64_t val = 0U;
        buffer >> val;

        std::bitset<Size> result(val);
        // bit_set(std::move(result));
        bit_set = result;
    }
};

}  // namespace ov::intel_cpu
