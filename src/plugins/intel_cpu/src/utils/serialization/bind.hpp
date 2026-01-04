// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>
#include <functional>
#include <stdarg.h>

#include "buffers.hpp"
//#include "graph_context.h"
#include "static_instance.hpp"
#include "onednn/dnnl.h"

// Description
// SaversStorage - keeps functions to save objects
// LoadesrStorage - keeps functions to load objects


#define DECLARE_SERIALIZATION_OBJECT_MEMBERS(cls_name)                                  \
    static const std::string& get_type_info_s() {                                       \
        static const std::string type_name(#cls_name);                                  \
        /*printf("[CPU] TYPE: '%s'\n", type_name.data()); /* TODO: remove */ \
        return type_name;                                                               \
    }                                                                                   \
    virtual const std::string& get_type_info() const { return get_type_info_s(); }

#define DECLARE_SERIALIZATION_OBJECT_MEMBERS_OVERRIDE(cls_name)                         \
    static const std::string& get_type_info_s() {                                       \
        static const std::string type_name = #cls_name;                                 \
        /*printf("[CPU] TYPE: '%s'\n", type_name.data()); /* TODO: remove */ \
        return type_name;                                                               \
    }                                                                                   \
    const std::string& get_type_info() const override { return get_type_info_s(); }

#define BIND_TO_BUFFER(Buffer, ObjType)                                                 \
    template <>                                                                         \
    class BindCreator<Buffer, ObjType> {                                                \
    private:                                                                            \
        static const InstanceCreator<Buffer, ObjType>& m_creator;                       \
    };                                                                                  \
    const InstanceCreator<Buffer, ObjType>& BindCreator<Buffer, ObjType>::m_creator =   \
        StaticInstance<InstanceCreator<Buffer, ObjType>>::get_instance().instantiate();


namespace ov::intel_cpu {

template <typename BufferType>
struct SaverStorage {
    using SaveFunction = std::function<void(BufferType&, const void*)>;
    using ValueType = typename std::unordered_map<size_t, SaveFunction>::value_type;

    static SaverStorage<BufferType>& instance() {
        static SaverStorage<BufferType> instance;
        return instance;
    }

    const SaveFunction& get_save_function(const size_t type_hash) const {
        auto it = m_functions_map.find(type_hash);
        OPENVINO_ASSERT(it != m_functions_map.end(), "[ SERIALIZER ] Could nod find save function for object '", type_hash, "'");
        return it->second;
    }

    void set_save_function(const ValueType& pair) {  // TODO: static?
        m_functions_map.insert(pair);
    }

private:
    SaverStorage() = default;
    SaverStorage(const SaverStorage&) = delete;
    void operator=(const SaverStorage&) = delete;

    std::unordered_map<size_t, SaveFunction> m_functions_map;  // TODO: static?
};
// #define ...
// SomeWrapper{
//     SaverStorage<BufferType>::set_save_function(pair);
// }

template <typename T>
struct VoidDeleter {
    void operator()(const T*) const { }
};

template <typename BufferType, typename FuncType>
struct LoaderStorage {
    using ValueType = typename std::unordered_map<size_t, FuncType>::value_type;

    static LoaderStorage& instance() {
        static LoaderStorage instance;
        return instance;
    }

    const FuncType& get_load_function(const size_t type_hash) {
        printf("[ SER ] LoaderStorage::get_load_function type: '%zu'\n", type_hash);  // TODO: remove
        auto it = m_functions_map.find(type_hash);
        OPENVINO_ASSERT(it != m_functions_map.end(), "[ SERIALIZER ] Could nod find load function for object '", type_hash, "'");
        return it->second;
    }

    void set_load_function(const ValueType& pair) {
        // if (m_functions_map.find(pair.first) == m_functions_map.end()) {  // TODO: remove
            printf("[ SER ] LoaderStorage::set_load_function '%zu'\n", pair.first);
        // }
        m_functions_map.insert(pair);  // TODO: Actually there is a one fn for different Keys. Change key to args list?
    }

private:
    LoaderStorage() = default;
    LoaderStorage(const LoaderStorage&) = delete;
    void operator=(const LoaderStorage&) = delete;

    std::unordered_map<size_t, FuncType> m_functions_map;
};

template <typename BufferType>
using def = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&)>>;

// template <typename BufferType>
// using VarLS = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, ...)>>;

// template <typename BufferType>
// using dif = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, dnnl::engine&)>>;

// template <typename BufferType>
// using dif = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, const ov::intel_cpu::GraphContext::Ptr&)>>;

class GraphContext;
template <typename BufferType>
using LSGraphContext = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, const std::shared_ptr<const GraphContext>&)>>;

// template <typename BufferType>
// using dif = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, ...)>>;

// TODO: Use initialization list instead of BindCreator
// #define SET_LOAD_FUNCTION() \
// LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&)>>::set_load_function()

template <typename BufferType, typename T, typename Enable = void>
class BufferBinder;

template <typename BufferType, typename T>
class BufferBinder<BufferType, T, typename std::enable_if_t<std::is_base_of_v<OutputBuffer<BufferType>, BufferType>>> {
public:
    static BufferBinder& instance() {
        static BufferBinder instance;
        return instance;
    }

private:
    BufferBinder() {
        SaverStorage<BufferType>::instance().set_save_function({typeid(T).hash_code(), save});
    }

    BufferBinder(const BufferBinder&) = delete;
    void operator=(const BufferBinder&) = delete;

    template <typename Derived>
    static const Derived* downcast(const void* base_ptr) {
        return static_cast<Derived const *>(base_ptr);
    }

    static void save(BufferType& buffer, const void* base_ptr) {
        const auto derived_ptr = downcast<T>(base_ptr);
        derived_ptr->save(buffer);
    }
};

template <typename BufferType, typename T>
class BufferBinder<BufferType, T, typename std::enable_if_t<std::is_base_of_v<InputBuffer<BufferType>, BufferType> &&
                                                            std::is_default_constructible<T>::value>> {
public:
    static BufferBinder& instance() {
        static BufferBinder instance;
        return instance;
    }

private:
    BufferBinder() {
        printf("[ SER ] BufferBinder '%s'\n", typeid(T).name());
        def<BufferType>::instance().set_load_function(
                {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr) {
            std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T());
            derived_ptr->load(buffer);
            dst_ptr.reset(derived_ptr.release());
        }});

    // BufferBinder() {
    //     VarLS<BufferType>::instance().set_load_function(
    //             {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr, ...) {
    //         std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T());
    //         va_list args;
    //         va_start(args, dst_ptr);
    //         derived_ptr->load(buffer, args);
    //         va_end(args);
    //         dst_ptr.reset(derived_ptr.release());
    //     }});
    }

    BufferBinder(const BufferBinder&) = delete;
    void operator=(const BufferBinder&) = delete;
};

template <typename BufferType, typename T>
class BufferBinder<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                          !std::is_default_constructible<T>::value>::type> {
public:
    static BufferBinder& instance() {
        static BufferBinder instance;
        return instance;
    }

private:
    BufferBinder() {
        printf("[ SER ] BufferBinder '%s'\n", typeid(T).name());
        // def<BufferType>::instance().set_load_function(  // TODO: it should for default ctr only. Remove.
        //         {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr) {
        //     std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T());
        //     derived_ptr->load(buffer);
        //     dst_ptr.reset(derived_ptr.release());
        // }});

        // dif<BufferType>::instance().set_load_function(
        //         {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr, dnnl::engine& engine) {
        //     std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(engine));
        //     derived_ptr->load(buffer);
        //     dst_ptr.reset(derived_ptr.release());
        // }});

        // dif<BufferType>::instance().set_load_function(
        //         {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr, const GraphContext::Ptr& context) {
        //     std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(context));
        //     derived_ptr->load(buffer);
        //     dst_ptr.reset(derived_ptr.release());
        // }});

        // dif<BufferType>::instance().set_load_function(
        //     {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr, const GraphContext::Ptr& context) {
        //     std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(buffer, context));
        //     dst_ptr.reset(derived_ptr.release());
        // }});

        LSGraphContext<BufferType>::instance().set_load_function(
            {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr, const GraphContext::CPtr& context) {
            std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(buffer, context));
            // auto derived_ptr = std::unique_ptr<T>(new T(buffer, context));
            dst_ptr.reset(derived_ptr.release());
        }});
    }

    BufferBinder(const BufferBinder&) = delete;
    void operator=(const BufferBinder&) = delete;
};

// template <typename BufferType, typename T>
// class BufferBinder<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
//                                                            !std::is_default_constructible<T>::value>::type> {
// public:
//     static BufferBinder& instance() {
//         static BufferBinder instance;
//         return instance;
//     }

// private:
//     BufferBinder() {
//         diif<BufferType>::instance().set_load_function(
//                 {typeid(T).hash_code(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& dst_ptr, ...) {
//             va_list a_list;
//             std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(a_list));
//             derived_ptr->load(buffer);
//             dst_ptr.reset(derived_ptr.release());
//         }});
//     }

//     BufferBinder(const BufferBinder&) = delete;
//     void operator=(const BufferBinder&) = delete;
// };

template <typename BufferType, typename T>
class BindCreator;

template <typename BufferType, typename T>
class InstanceCreator {
public:
    const InstanceCreator& instantiate() {
        StaticInstance<BufferBinder<BufferType, T>>::get_instance();
        return *this;
    }
};

}  // namespace ov::intel_cpu
