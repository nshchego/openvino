// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <type_traits>
#include <utility>
#include <iostream>

namespace ov::intel_cpu {

template <typename T, typename Enable = void>
class StaticInstance;

template <typename T>
class StaticInstance<T, typename std::enable_if<std::is_default_constructible<T>::value>::type> {
public:
    static T& get_instance() {
        return instantiate();
    }

private:
    static T& instantiate() {
        static T singleton;
        (void)m_instance;
        return singleton;
    }

    static const T& m_instance;
};

template <typename T>
const T& StaticInstance<T, typename std::enable_if<std::is_default_constructible<T>::value>::type>::m_instance = StaticInstance<T>::instantiate();

template <typename T>
class StaticInstance<T, typename std::enable_if<!std::is_default_constructible<T>::value>::type> {
public:
    static T& get_instance() {
        return instantiate();
    }

private:
    static T& instantiate() {
        (void)m_instance;
        return T::instance();
    }

    static const T& m_instance;
};

template <typename T>
const T& StaticInstance<T, typename std::enable_if<!std::is_default_constructible<T>::value>::type>::m_instance = StaticInstance<T>::instantiate();

}  // namespace ov::intel_cpu
