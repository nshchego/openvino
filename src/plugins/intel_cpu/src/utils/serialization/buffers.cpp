// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "buffers.hpp"

#include "openvino/runtime/shared_buffer.hpp"

namespace ov::intel_cpu {

////////// BinaryOutputBuffer //////////

BinaryOutputBuffer::BinaryOutputBuffer(std::ostream& stream)
    : OutputBuffer<BinaryOutputBuffer>(this), m_stream(stream), m_impl_params(nullptr), m_strm(nullptr) {
    m_header_offset = m_stream.tellp();
}

void BinaryOutputBuffer::dump_position() {
    // *this << size_t(m_stream.tellp()) - m_header_offset;
    auto tp = m_stream.tellp();
    size_t off = size_t(tp) - m_header_offset;
    *this << off;
}

////////// BinaryInputBuffer //////////

BinaryInputBuffer::BinaryInputBuffer(std::streampos header_offset)
    : InputBuffer<BinaryInputBuffer>(this), m_header_offset(header_offset) {}

// BinaryInputBuffer::BinaryInputBuffer(std::istream& stream)
//     : InputBuffer<BinaryInputBuffer>(this), m_stream(stream), m_impl_params(nullptr) {}

// void BinaryInputBuffer::read(void* const data, std::streamsize size) {
//     auto const read_size = m_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);  // TODO: get raw ptr instead of copy
//     OPENVINO_ASSERT(read_size == size,
//         "[ CPU ] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
// }

// size_t BinaryInputBuffer::get_position() const {
//     return m_stream.tellg();
// }

// const std::streambuf* BinaryInputBuffer::rdbuf() {
//     return m_stream.rdbuf();
// }

// std::istream& BinaryInputBuffer::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
//     return m_stream.seekg(offset, way);
// }

// template <>
// BinaryInputBuffer<std::istream>::BinaryInputBuffer(std::istream& stream)
//     : InputBuffer<BinaryInputBuffer>(this), m_stream(stream), m_impl_params(nullptr) {}

// template <>
// void BinaryInputBuffer<std::istream>::read(void* const data, std::streamsize size) {
//     auto const read_size = m_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);  // TODO: get raw ptr instead of copy
//     OPENVINO_ASSERT(read_size == size,
//         "[ CPU ] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
// }

// template <>
// size_t BinaryInputBuffer<std::istream>::get_position() const {
//     return m_stream.tellg();
// }

// template <>
// const std::streambuf* BinaryInputBuffer<std::istream>::rdbuf() {
//     return m_stream.rdbuf();
// }

// template <>
// std::istream& BinaryInputBuffer<std::istream>::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
//     return m_stream.seekg(offset, way);
// }

void BinaryInputBuffer::read(void* const data, std::streamsize size) {
    OPENVINO_THROW("Not implemented.");
}

void BinaryInputBuffer::read(const void*& data, std::streamsize size) {
    OPENVINO_THROW("Not implemented.");
}

void BinaryInputBuffer::check_position() {
    const auto act_pos = get_position();
    size_t exp_pos = 0lu;
    read(&exp_pos, sizeof(exp_pos));

if (exp_pos != act_pos) {  // TODO: remove
    printf("[ ERROR ] Invalid input stream position. Expected: %llu; Actual: %llu\n",
            exp_pos, act_pos);
}
    OPENVINO_ASSERT(exp_pos == act_pos,
                    "Invalid input stream position. Expected: ",
                    exp_pos,
                    "; Actual: ",
                    act_pos);
}

////////// StreamInputBuffer //////////

StreamInputBuffer::StreamInputBuffer(std::istream& stream, std::streampos header_offset)
    : BinaryInputBuffer(header_offset), m_stream(stream) {}

void StreamInputBuffer::read(void* const data, std::streamsize size) {
    auto const read_size = m_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);  // TODO: get raw ptr instead of copy
    OPENVINO_ASSERT(read_size == size,
        "[ CPU ] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
}

size_t StreamInputBuffer::get_position() const {
    OPENVINO_ASSERT(m_stream.tellg() >= m_header_offset, "StreamInputBuffer has incorrect offset.");
    return m_stream.tellg() - m_header_offset;
}

// void StreamInputBuffer::check_position() {
// }

// const std::streambuf* StreamInputBuffer::rdbuf() {
//     return m_stream.rdbuf();
// }
const void* StreamInputBuffer::get_data() {
    auto buff = dynamic_cast<const SharedStreamBuffer*>(m_stream.rdbuf());
    OPENVINO_ASSERT(buff, "got unexpected input buffer type.");

    return buff->get_data();
}

// std::istream& StreamInputBuffer::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
//     return m_stream.seekg(offset, way);
// }
void StreamInputBuffer::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
    m_stream.seekg(offset, way);
}

////////// TensorInputBuffer //////////

TensorInputBuffer::TensorInputBuffer(const ov::Tensor& model_tensor, std::streampos header_offset)
    : BinaryInputBuffer(header_offset), m_model_tensor(model_tensor) {
    m_data = static_cast<const uint8_t *>(m_model_tensor.data());
}

void TensorInputBuffer::read(void* const data, std::streamsize size) {
    memcpy(data, m_data + m_offset, size);
    m_offset += size;
}

void TensorInputBuffer::read(const void*& data, std::streamsize size) {
    data = m_data + m_offset;
    m_offset += size;
}

size_t TensorInputBuffer::get_position() const {
    OPENVINO_ASSERT(m_offset >= m_header_offset, "TensorInputBuffer has incorrect offset.");
    return m_offset - m_header_offset;
}

// void TensorInputBuffer::check_position() {
//     const auto act_pos = get_position();
//     size_t exp_pos = 0lu;
//     read(&exp_pos, sizeof(exp_pos));

// if (exp_pos != act_pos) {  // TODO: remove
//     printf("[ ERROR ] Invalid input stream position. Expected: %llu; Actual: %llu\n",
//             exp_pos, act_pos);
// }
//     OPENVINO_ASSERT(exp_pos == act_pos,
//                     "Invalid input stream position. Expected: ",
//                     exp_pos,
//                     "; Actual: ",
//                     act_pos);
// }

// const std::streambuf* TensorInputBuffer::rdbuf() {
//     return nullptr;
// }
const void* TensorInputBuffer::get_data() {
    return m_data;
}

// std::istream& TensorInputBuffer::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
//     return m_offset = offset;
// }
void TensorInputBuffer::seekg(std::istream::off_type offset, std::ios_base::seekdir way) {
    // TODO: check bounds
    if (way == std::ios_base::cur) {
        m_offset += offset; 
    } else if (way == std::ios_base::beg) {
        m_offset = offset;
    } else if (way == std::ios_base::end) {
        // m_offset = len - offset;
    } else {
        OPENVINO_THROW("...");
    }
}

}  // namespace ov::intel_cpu
