// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <utility>
#include <type_traits>

namespace ov::intel_cpu {

template <typename T>
struct Data {
    using DataType = typename std::conditional<std::is_const<typename std::remove_pointer<typename std::remove_reference<T>::type>::type>::value,
                                               const void*, void*>::type;

    Data(T&& data, uint64_t number_of_bytes) : m_data(std::forward<T>(data)), m_number_of_bytes(number_of_bytes) {}

    DataType m_data;
    uint64_t m_number_of_bytes;
};

template <typename T>
static Data<T> make_data(T&& data, uint64_t number_of_bytes) {
    return {std::forward<T>(data), number_of_bytes};
}

}  // namespace ov::intel_cpu
