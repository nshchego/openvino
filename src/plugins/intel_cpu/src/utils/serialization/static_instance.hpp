// Copyright (C) 2018-2025 Intel Corporation
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
        (void)instance;
        return singleton;
    }

    static const T& instance;
};

template <typename T>
const T& StaticInstance<T, typename std::enable_if<std::is_default_constructible<T>::value>::type>::instance = StaticInstance<T>::instantiate();

template <typename T>
class StaticInstance<T, typename std::enable_if<!std::is_default_constructible<T>::value>::type> {
public:
    static T& get_instance() {
        return instantiate();
    }

private:
    static T& instantiate() {
        (void)instance;
        return T::instance();
    }

    static const T& instance;
};

template <typename T>
const T& StaticInstance<T, typename std::enable_if<!std::is_default_constructible<T>::value>::type>::instance = StaticInstance<T>::instantiate();

}  // namespace ov::intel_cpu
