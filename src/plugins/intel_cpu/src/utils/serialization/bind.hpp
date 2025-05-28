// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <unordered_map>
#include <functional>

#include "buffer.hpp"
#include "static_instance.hpp"


#define DECLARE_OBJECT_TYPE_SERIALIZATION(cls_name)                                   \
    static const std::string& get_type_info_s() {                                     \
        static const std::string type_name = #cls_name;                               \
        return type_name;                                                             \
    }                                                                                 \
    const std::string& get_type_info() const override { return get_type_info_s(); }

#define BIND_TO_BUFFER(buffer, type)                                                  \
    template <>                                                                       \
    class BindCreator<buffer, type> {                                                 \
    private:                                                                          \
        static const InstanceCreator<buffer, type>& creator;                          \
    };                                                                                \
    const InstanceCreator<buffer, type>& BindCreator<buffer, type>::creator =         \
        StaticInstance<InstanceCreator<buffer, type>>::get_instance().instantiate();


namespace ov::intel_cpu {

template <typename BufferType>
struct SaverStorage {
    using save_function = std::function<void(BufferType&, const void*)>;
    using value_type = typename std::unordered_map<std::string, save_function>::value_type;

    static SaverStorage<BufferType>& instance() {
        static SaverStorage<BufferType> instance;
        return instance;
    }

    const save_function& get_save_function(const std::string& type) const {
        return m_map.at(type);
    }

    void set_save_function(const value_type& pair) {
        m_map.insert(pair);
    }

private:
    SaverStorage() = default;
    SaverStorage(const SaverStorage&) = delete;
    void operator=(const SaverStorage&) = delete;

    std::unordered_map<std::string, save_function> m_map;
};

template <typename T>
struct VoidDeleter {
    void operator()(const T*) const { }
};

template <typename BufferType, typename FuncT>
struct LoaderStorage {
    using value_type = typename std::unordered_map<std::string, FuncT>::value_type;

    static LoaderStorage& instance() {
        static LoaderStorage instance;
        return instance;
    }

    const FuncT& get_load_function(const std::string& type) {
        return map.at(type);
    }

    void set_load_function(const value_type& pair) {
        map.insert(pair);
    }

private:
    LoaderStorage() = default;
    LoaderStorage(const LoaderStorage&) = delete;
    void operator=(const LoaderStorage&) = delete;

    std::unordered_map<std::string, FuncT> map;
};

template <typename BufferType>
using def = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&)>>;

template <typename BufferType>
using dif = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, dnnl::engine&)>>;

// template <typename BufferType>
// using diif = LoaderStorage<BufferType, std::function<void(BufferType&, std::unique_ptr<void, VoidDeleter<void>>&, ...)>>;

template <typename BufferType, typename T, typename Enable = void>
class BufferBinder;

template <typename BufferType, typename T>
class BufferBinder<BufferType, T, typename std::enable_if<std::is_base_of<OutputBuffer<BufferType>, BufferType>::value>::type> {
public:
    static BufferBinder& instance() {
        static BufferBinder instance;
        return instance;
    }

private:
    BufferBinder() {
        SaverStorage<BufferType>::instance().set_save_function({T::get_type_info_s(), save});
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
class BufferBinder<BufferType, T, typename std::enable_if<std::is_base_of<InputBuffer<BufferType>, BufferType>::value &&
                                                           std::is_default_constructible<T>::value>::type> {
public:
    static BufferBinder& instance() {
        static BufferBinder instance;
        return instance;
    }

private:
    BufferBinder() {
        def<BufferType>::instance().set_load_function(
                {T::get_type_info_s(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& result_ptr) {
            std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T());
            derived_ptr->load(buffer);
            result_ptr.reset(derived_ptr.release());
        }});
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
        dif<BufferType>::instance().set_load_function(
                {T::get_type_info_s(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& result_ptr, dnnl::engine& engine) {
            std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(engine));
            derived_ptr->load(buffer);
            result_ptr.reset(derived_ptr.release());
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
//                 {T::get_type_info_s(), [](BufferType& buffer, std::unique_ptr<void, VoidDeleter<void>>& result_ptr, ...) {
//             va_list a_list;
//             std::unique_ptr<T> derived_ptr = std::unique_ptr<T>(new T(a_list));
//             derived_ptr->load(buffer);
//             result_ptr.reset(derived_ptr.release());
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
