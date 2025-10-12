// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>

#include "buffers.hpp"

namespace ov::intel_cpu {

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::map<Key, Value>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::map<Key, Value>& map) {
// printf("-WRITE map-\n");
        buffer << map.size();
        for (const auto& pair : map) {
            buffer(pair.first, pair.second);
        }
    }
};

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::map<Key, Value>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::map<Key, Value>& map) {
// printf("-READ map-\n");
        typename std::map<Key, Value>::size_type map_size = 0UL;
        buffer >> map_size;
        map.clear();
        Key key;
        for (size_t i = 0; i < map_size; i++) {
            buffer >> key;
            buffer >> map[std::move(key)];
        }
    }
};


template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::unordered_map<Key, Value>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::unordered_map<Key, Value>& map) {
// printf("-WRITE unordered_map-\n");
        const auto map_size = map.size();
        buffer.write(std::addressof(map_size), sizeof(map_size));
        for (const auto& pair : map) {
            buffer(pair.first, pair.second);
        }
    }
};

template <typename BufferType, typename Key, typename Value>
class Serializer<BufferType, std::unordered_map<Key, Value>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::unordered_map<Key, Value>& map) {
// printf("-READ unordered_map-\n");
        typename std::unordered_map<Key, Value>::size_type map_size = 0UL;
        buffer.read(std::addressof(map_size), sizeof(map_size));
        map.clear();
        Key key;
        for (size_t i = 0; i < map_size; i++) {
            buffer >> key;
            buffer >> map[std::move(key)];
        }
    }
};

}  // namespace ov::intel_cpu
