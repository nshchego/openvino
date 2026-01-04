// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <exception>
#include <type_traits>

// #include "../buffers.hpp"
#include "../bind.hpp"
#include "graph_context.h"
// #include "../helpers.hpp"

namespace ov {
namespace intel_cpu {

// template <typename BufferType>
// class BufferBinder<BufferType, std::any, typename std::enable_if_t<std::is_base_of_v<OutputBuffer<BufferType>, BufferType>>> {
// public:
//     static BufferBinder& instance() {
//         static BufferBinder instance;
//         return instance;
//     }

// private:
//     BufferBinder() {
//         SaverStorage<BufferType>::instance().set_save_function({typeid(T).hash_code(), save});
//     }

//     BufferBinder(const BufferBinder&) = delete;
//     void operator=(const BufferBinder&) = delete;

//     template <typename Derived>
//     static const Derived* downcast(const void* base_ptr) {
//         return static_cast<Derived const *>(base_ptr);
//     }

//     static void save(BufferType& buffer, const void* base_ptr) {
//         const auto derived_ptr = downcast<T>(base_ptr);
//         derived_ptr->save(buffer);
//     }
// };

template <typename BufferType>
class Serializer<BufferType, std::any, typename std::enable_if_t<std::is_base_of_v<OutputBuffer<BufferType>, BufferType>>> {
public:
    static void save(BufferType& buffer, const std::any& src) {
        // buffer << str.size();
        // buffer << make_data(str.data(), static_cast<uint64_t>(str.size() * sizeof(std::any::value_type)));

        // const size_t type_hash = src.type().hash_code();
        // // printf("[ SER ] Serializer::save type: '%s'\n", typeid(*src_ptr.get()).name());  // TODO: remove
        // buffer << type_hash;
        // const auto save_func = SaverStorage<BufferType>::instance().get_save_function(type_hash);
        // save_func(buffer, src);

        buffer << src.type().hash_code();
        if (src.type() == typeid(ActivationPostOp)) {
            buffer << std::any_cast<ActivationPostOp>(src);
        } else if (src.type() == typeid(ScaleShiftPostOp)) {
            buffer << std::any_cast<ScaleShiftPostOp>(src);
        } else if (src.type() == typeid(FakeQuantizePostOp)) {
            buffer << std::any_cast<FakeQuantizePostOp>(src);
        } else if (src.type() == typeid(DepthwiseConvolutionPostOp)) {
            buffer << std::any_cast<DepthwiseConvolutionPostOp>(src);
        } else {
            OPENVINO_THROW("[ SERIALIZATION ] Could not serialize object '", src.type().name(), "'");
        }
    }
};

template <typename BufferType>
class Serializer<BufferType, std::any, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static void load(BufferType& buffer, std::any& dst) {
        size_t type_hash;
        buffer >> type_hash;
        if (type_hash == typeid(ActivationPostOp).hash_code()) {
            ActivationPostOp obj;
            buffer >> obj;
            dst = obj;
        } else if (type_hash == typeid(ScaleShiftPostOp).hash_code()) {
            ScaleShiftPostOp obj;
            buffer >> obj;
            dst = obj;
        } else if (type_hash == typeid(FakeQuantizePostOp).hash_code()) {
            FakeQuantizePostOp obj;
            buffer >> obj;
            dst = obj;
        } else if (type_hash == typeid(DepthwiseConvolutionPostOp).hash_code()) {
            DepthwiseConvolutionPostOp obj;
            buffer >> obj;
            dst = obj;
        } else {
            OPENVINO_THROW("[ SERIALIZATION ] Could not deserialize object '", type_hash, "'");
        }
    }
};

}  // namespace intel_cpu
}  // namespace ov
