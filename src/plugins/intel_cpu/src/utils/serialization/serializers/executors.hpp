// Copyright (C) 2018-2026 Intel Corporation
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

namespace ov {
namespace intel_cpu {

// template <typename BufferType, typename T>
// class Serializer<BufferType, std::shared_ptr<T>, typename std::enable_if<std::is_base_of_v<OutputBuffer<BufferType>, BufferType> && std::is_base_of_v<Executor, T>>::type> {
// public:
// //     static void save(BufferType& buffer, const std::shared_ptr<T>& ptr) {
// // // printf("-WRITE shared_ptr- '%s'\n", typeid(*(ptr.get())).name());
// //         // const char* type = typeid(*(ptr.get())).name();
// //         // buffer << type;
// //         // if (strcmp(type, "NONE") != 0) {
// //         const std::string type(typeid(*(ptr.get())).name());
// //         // const std::string& type = ptr->get_type_info();  // TODO: replace with char*
// //         if (type.compare("NONE") != 0) {
// //             const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type);
// //             save_func(buffer, ptr.get());
// //         }
// //     }

//     static void save(BufferType& buffer, const std::shared_ptr<T>& ptr) {
//         const uint8_t not_null = ptr != nullptr ? 1 : 0;
//         buffer << not_null;
//         if (not_null == 0) {
//             return;
//         }

//         const std::string& type = ptr->get_type_info();
//         buffer << type;
//         if (type.compare("NONE") != 0) {
//             const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type);
//             save_func(buffer, ptr.get());
//         }
//     }
// };

template <typename BufferType, typename ObjT, typename AttrT>
class Serializer<BufferType, std::shared_ptr<ObjT>, typename std::enable_if<std::is_base_of_v<InputBuffer<BufferType>, BufferType> && std::is_base_of_v<Executor, ObjT>>::type> {
public:
//     static void load(BufferType& buffer, std::shared_ptr<ObjT>& ptr, dnnl::engine& engine) {
// // printf("-READ shared_ptr eng-\n");
//         std::string type;
//         buffer >> type;
//         if (type.compare("NONE") != 0) {
//             const auto load_func = dif<BufferType>::instance().get_load_function(type);
//             std::unique_ptr<void, VoidDeleter<void>> result;
//             load_func(buffer, result, engine);
//             ptr.reset(static_cast<ObjT*>(result.release()));
//         }
//     }

    static void load(BufferType& buffer, std::shared_ptr<ObjT>& ptr, ...) {
        uint8_t not_null;
        buffer >> not_null;
        if (not_null == 0) {
            ptr = nullptr;
            return;
        }

        std::string type;
        buffer >> type;
        if (type.compare("NONE") != 0) {
            const auto load_func = dif<BufferType>::instance().get_load_function(type);
            std::unique_ptr<void, VoidDeleter<void>> result;
            load_func(buffer, result, context);
            ptr.reset(static_cast<ObjT*>(result.release()));
        }
    }

//     static void load(BufferType& buffer, std::shared_ptr<ObjT>& ptr) {
// // printf("-READ shared_ptr-\n");
//         uint8_t not_null;
//         buffer >> not_null;
//         if (not_null == 0) {
//             ptr = nullptr;
//             return;
//         }

//         std::string type;
//         buffer >> type;
//         if (type.compare("NONE") != 0) {
//             const auto load_func = def<BufferType>::instance().get_load_function(type);
//             std::unique_ptr<void, VoidDeleter<void>> result;
//             load_func(buffer, result);
//             ptr.reset(static_cast<ObjT*>(result.release()));
//         }
//     }
};

}  // namespace intel_cpu
}  // namespace ov
