// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include "buffers.hpp"
#include "helpers.hpp"

namespace ov::intel_cpu {

template <typename BufferType>
class Serializer<BufferType, std::string, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::string& str) {
printf("-WRITE string-\n");
        buffer << str.size();
        buffer << make_data(str.data(), static_cast<uint64_t>(str.size() * sizeof(std::string::value_type)));
    }
};

template <typename BufferType>
class Serializer<BufferType, std::string, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::string& str) {
// printf("-READ string-\n");
        std::string::size_type size;
        buffer >> size;
        str.resize(size);
        buffer >> make_data(const_cast<std::string::value_type*>(str.data()), static_cast<uint64_t>(size * sizeof(std::string::value_type)));
    }
};

// TODO: Reuse char* for get_type_info
// template <typename BufferType>
// class Serializer<BufferType, char*, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
// public:
//     static void save(BufferType& buffer, const char* str) {
// printf("-WRITE char*-\n");
//         buffer << make_data(str, strlen(str));
//     }
// };

// template <typename BufferType>
// class Serializer<BufferType, char*, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
// public:
//     static void load(BufferType& buffer, const char* str) {
// // printf("-READ char*-\n");
//         // std::string::size_type size;
//         // buffer >> size;
//         // str.resize(size);
//         // buffer >> make_data(const_cast<std::string::value_type*>(str.data()), static_cast<uint64_t>(size * sizeof(std::string::value_type)));
//     }
// };

}  // namespace ov::intel_cpu
