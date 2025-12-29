// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// #include <set>
#include <type_traits>
// #include <unordered_set>

#include "../buffers.hpp"
// #include "../helpers.hpp"

namespace ov::intel_cpu {

template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value && std::is_array_v<T>>::type> {
public:
    static void save(BufferType& buffer, const T& arr) {
// printf("-WRITE enum-\n");  // TODO: remove
// if (sizeof(T) > 1UL) {
//     printf("-WRITE enum- size: %llu\n", sizeof(T));  // TODO: Move enums to uint8 if possible.
// }
        buffer.write(std::addressof(arr), sizeof(arr));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value && std::is_array_v<T>>::type> {
public:
    static void load(BufferType& buffer, T& arr) {
// printf("-READ enum-\n");  // TODO: remove
        buffer.read(std::addressof(arr), sizeof(arr));
    }
};

// template <typename BufferType, typename T>
// class Serializer<BufferType, std::set<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
// public:
//     static void save(BufferType& buffer, const std::set<T>& set) {
//         buffer << set.size();
//         for (const auto& el : set) {
//             buffer << el;
//         }
//     }
// };

// template <typename BufferType, typename T>
// class Serializer<BufferType, std::set<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
// public:
//     static void load(BufferType& buffer, std::set<T>& set) {
//         typename std::set<T>::size_type set_size = 0UL;
//         buffer >> set_size;

//         for (size_t i = 0UL; i < set_size; i++) {
//             T el;
//             buffer >> el;
//             set.insert(el);
//         }
//     }
// };

// template <typename BufferType, typename T>
// class Serializer<BufferType, std::unordered_set<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
// public:
//     static void save(BufferType& buffer, const std::unordered_set<T>& set) {
//         buffer << set.size();
//         for (const auto& el : set) {
//             buffer << el;
//         }
//     }
// };

// template <typename BufferType, typename T>
// class Serializer<BufferType, std::unordered_set<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
// public:
//     static void load(BufferType& buffer, std::unordered_set<T>& set) {
//         typename std::unordered_set<T>::size_type set_size = 0UL;
//         buffer >> set_size;
//         set.reserve(set_size);
//         for (size_t i = 0UL; i < set_size; i++) {
//             T el;
//             buffer >> el;
//             set.insert(el);
//         }
//     }
// };

}  // namespace ov::intel_cpu
