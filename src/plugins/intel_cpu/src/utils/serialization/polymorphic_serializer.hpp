// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <exception>
#include <type_traits>

#include "buffers.hpp"
#include "bind.hpp"
#include "graph_context.h"
#include "helpers.hpp"
//#include "onednn/dnnl.h"

namespace ov {
namespace intel_cpu {

template <typename BufferType, typename T>
class Serializer<BufferType, std::unique_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::unique_ptr<T>& ptr) {
printf("-WRITE unique_ptr-\n");
        const auto& type = ptr->get_type_info();
        buffer << type;
        const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type);
        save_func(buffer, ptr.get());
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::unique_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::unique_ptr<T>& ptr, dnnl::engine& engine) {
printf("-READ unique_ptr eng-\n");
        std::string type;
        buffer >> type;
        const auto load_func = dif<BufferType>::instance().get_load_function(type);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result, engine);
        ptr.reset(static_cast<T*>(result.release()));
    }

    static void load(BufferType& buffer, std::unique_ptr<T>& ptr) {
printf("-READ unique_ptr-\n");
        std::string type;
        buffer >> type;
        const auto load_func = def<BufferType>::instance().get_load_function(type);
        std::unique_ptr<void, VoidDeleter<void>> result;
        load_func(buffer, result);
        ptr.reset(static_cast<T*>(result.release()));
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::shared_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
//     static void save(BufferType& buffer, const std::shared_ptr<T>& ptr) {
// // printf("-WRITE shared_ptr- '%s'\n", typeid(*(ptr.get())).name());
//         // const char* type = typeid(*(ptr.get())).name();
//         // buffer << type;
//         // if (strcmp(type, "NONE") != 0) {
//         const std::string type(typeid(*(ptr.get())).name());
//         // const std::string& type = ptr->get_type_info();  // TODO: replace with char*
//         if (type.compare("NONE") != 0) {
//             const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type);
//             save_func(buffer, ptr.get());
//         }
//     }

    static void save(BufferType& buffer, const std::shared_ptr<T>& ptr) {
// printf("-WRITE shared_ptr- '%s'\n", typeid(*(ptr.get())).name());
        const uint8_t not_null = ptr != nullptr ? 1 : 0;
        buffer << not_null;
        if (not_null == 0) {
            return;
        }

        //buffer.dump_position();  // TODO: remove
        const std::string& type = ptr->get_type_info();
        buffer << type;
        if (type.compare("NONE") != 0) {
            const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type);
            save_func(buffer, ptr.get());
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::shared_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
//     static void load(BufferType& buffer, std::shared_ptr<T>& ptr, dnnl::engine& engine) {
// // printf("-READ shared_ptr eng-\n");
//         std::string type;
//         buffer >> type;
//         if (type.compare("NONE") != 0) {
//             const auto load_func = dif<BufferType>::instance().get_load_function(type);
//             std::unique_ptr<void, VoidDeleter<void>> result;
//             load_func(buffer, result, engine);
//             ptr.reset(static_cast<T*>(result.release()));
//         }
//     }

    static void load(BufferType& buffer, std::shared_ptr<T>& ptr, const GraphContext::CPtr& context) {
// printf("-READ shared_ptr eng-\n");
        uint8_t not_null;
        buffer >> not_null;
        if (not_null == 0) {
            ptr = nullptr;
            return;
        }

        //buffer.check_position();  // TODO: Remove
        std::string type;
        buffer >> type;
        if (type.compare("NONE") != 0) {
            const auto load_func = dif<BufferType>::instance().get_load_function(type);
            std::unique_ptr<void, VoidDeleter<void>> result;
            load_func(buffer, result, context);
            ptr.reset(static_cast<T*>(result.release()));
        }
    }

    static void load(BufferType& buffer, std::shared_ptr<T>& ptr) {
// printf("-READ shared_ptr-\n");
        uint8_t not_null;
        buffer >> not_null;
        if (not_null == 0) {
            ptr = nullptr;
            return;
        }

        std::string type;
        buffer >> type;
        if (type.compare("NONE") != 0) {
            const auto load_func = def<BufferType>::instance().get_load_function(type);
            std::unique_ptr<void, VoidDeleter<void>> result;
            load_func(buffer, result);
            ptr.reset(static_cast<T*>(result.release()));
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::weak_ptr<T>, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void save(BufferType& buffer, const std::weak_ptr<T>& ptr) {
printf("-WRITE weak_ptr-\n");
        if (auto shared = ptr.lock()) {
            shared->save(buffer);
        }
    }
};

template <typename BufferType, typename T>
class Serializer<BufferType, std::weak_ptr<T>, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::weak_ptr<T>& ptr, dnnl::engine& engine) {
printf("-READ weak_ptr eng-\n");
        std::string type;
        buffer >> type;
        if (type.compare("NONE") != 0) {
            const auto load_func = dif<BufferType>::instance().get_load_function(type);
            std::unique_ptr<void, VoidDeleter<void>> result;
            load_func(buffer, result, engine);
            ptr.reset(static_cast<T*>(result.release()));
        }
    }

    static void load(BufferType& buffer, std::weak_ptr<T>& ptr) {
printf("-READ weak_ptr-\n");
        std::string type;
        buffer >> type;
        if (type.compare("NONE") != 0) {
            const auto load_func = def<BufferType>::instance().get_load_function(type);
            std::unique_ptr<void, VoidDeleter<void>> result;
            load_func(buffer, result);
            ptr.reset(static_cast<T*>(result.release()));
        }
    }
};

}  // namespace intel_cpu
}  // namespace ov
