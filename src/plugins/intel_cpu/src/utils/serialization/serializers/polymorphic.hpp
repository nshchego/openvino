// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <exception>
#include <type_traits>

#include "../buffers.hpp"
#include "graph_context.h"

namespace ov::intel_cpu {

template <typename BufferType, typename T>
class Serializer<BufferType, std::unique_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::unique_ptr<T>& src_ptr) {
        const size_t type_hash = typeid(*src_ptr.get()).hash_code();
        // printf("[ SER ] Serializer::save type: '%s'\n", typeid(*src_ptr.get()).name());  // TODO: remove
        buffer << type_hash;
        const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type_hash);
        save_func(buffer, src_ptr.get());
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::unique_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::unique_ptr<T>& dst_ptr, dnnl::engine& engine) {
        size_t type_hash;
        buffer >> type_hash;
        const auto load_func = LSGraphContext<BufferType>::instance().get_load_function(type_hash);
        std::unique_ptr<void, VoidDeleter<void>> result;  // TODO: use void* instead?
        load_func(buffer, result, engine);
        dst_ptr.reset(static_cast<T*>(result.release()));
    }

    static void load(BufferType& buffer, std::unique_ptr<T>& dst_ptr) {
        size_t type_hash;
        buffer >> type_hash;
        const auto load_func = def<BufferType>::instance().get_load_function(type_hash);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result);
        dst_ptr.reset(static_cast<T*>(result.release()));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::shared_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
//     static void save(BufferType& buffer, const std::shared_ptr<T>& src_ptr) {
//         // const char* type = typeid(*(src_ptr.get())).name();
//         // buffer << type;
//         // if (strcmp(type, "NONE") != 0) {
//         const std::string type(typeid(*(src_ptr.get())).name());
//         // const std::string& type = src_ptr->get_type_info();  // TODO: replace with char*
//         if (type.compare("NONE") != 0) {
//             const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type);
//             save_func(buffer, src_ptr.get());
//         }
//     }

    static void save(BufferType& buffer, const std::shared_ptr<T>& src_ptr) {
        const uint8_t not_null = src_ptr != nullptr ? 1 : 0;
        buffer << not_null;
        if (not_null == 0) {
            return;
        }

        const size_t type_hash = typeid(*src_ptr.get()).hash_code();
        // printf("[ SER ] Serializer::save type: '%s'\n", typeid(*src_ptr.get()).name());  // TODO: remove
        buffer << type_hash;
        const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type_hash);
        save_func(buffer, src_ptr.get());
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::shared_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
//     static void load(BufferType& buffer, std::shared_ptr<T>& dst_ptr, dnnl::engine& engine) {
//         std::string type;
//         buffer >> type;
//         if (type.compare("NONE") != 0) {
//             const auto load_func = LSGraphContext<BufferType>::instance().get_load_function(type);
//             std::unique_ptr<void, VoidDeleter<void>> result;
//             load_func(buffer, result, engine);
//             dst_ptr.reset(static_cast<T*>(result.release()));
//         }
//     }

    static void load(BufferType& buffer, std::shared_ptr<T>& dst_ptr, const GraphContext::CPtr& context) {
        uint8_t not_null;
        buffer >> not_null;
        if (not_null == 0) {
            dst_ptr = nullptr;
            return;
        }

        size_t type_hash;
        buffer >> type_hash;
        const auto load_func = LSGraphContext<BufferType>::instance().get_load_function(type_hash);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result, context);
        dst_ptr.reset(static_cast<T*>(result.release()));
    }

    static void load(BufferType& buffer, std::shared_ptr<T>& dst_ptr) {
        uint8_t not_null;
        buffer >> not_null;
        if (not_null == 0) {
            dst_ptr = nullptr;
            return;
        }

        size_t type_hash;
        buffer >> type_hash;
        const auto load_func = def<BufferType>::instance().get_load_function(type_hash);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result);
        dst_ptr.reset(static_cast<T*>(result.release()));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::weak_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::weak_ptr<T>& src_ptr) {
        if (auto shared = src_ptr.lock()) {
            shared->save(buffer);
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::weak_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::weak_ptr<T>& dst_ptr, dnnl::engine& engine) {
        size_t type_hash;
        buffer >> type_hash;
        const auto load_func = LSGraphContext<BufferType>::instance().get_load_function(type_hash);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result, engine);
        dst_ptr.reset(static_cast<T*>(result.release()));
    }

    static void load(BufferType& buffer, std::weak_ptr<T>& dst_ptr) {
        size_t type_hash;
        buffer >> type_hash;
        const auto load_func = def<BufferType>::instance().get_load_function(type_hash);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result);
        dst_ptr.reset(static_cast<T*>(result.release()));
    }
};

}  // namespace ov::intel_cpu
