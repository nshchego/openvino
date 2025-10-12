// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffers.hpp"

namespace ov::intel_cpu {

BinaryInputBuffer::BinaryInputBuffer(std::istream& stream)
    : InputBuffer<BinaryInputBuffer>(this), m_stream(stream), m_impl_params(nullptr) {}

void BinaryInputBuffer::read(void* const data, std::streamsize size) {
    auto const read_size = m_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);  // TODO: get raw ptr instead of copy
    OPENVINO_ASSERT(read_size == size,
        "[ CPU ] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
}

size_t BinaryInputBuffer::get_pos() const {
    return m_stream.tellg();
}

const std::streambuf* BinaryInputBuffer::rdbuf() {
    return m_stream.rdbuf();
}

std::istream& BinaryInputBuffer::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
    return m_stream.seekg(offset, way);
}

}  // namespace ov::intel_cpu
