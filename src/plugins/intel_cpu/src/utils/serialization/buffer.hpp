// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>


#include "helpers.hpp"
#include "onednn/dnnl.h"
#include "openvino/core/except.hpp"
#include "serializer.hpp"

namespace ov::intel_cpu {

template <typename BufferType>
class Buffer {
public:
    Buffer(BufferType* const buffer) : buffer(buffer) {}

    template <typename ... Types>
    inline BufferType& operator()(Types&& ... args) {
        process(std::forward<Types>(args)...);
        return *buffer;
    }

protected:
    inline BufferType& getBuffer() {
        return *buffer;
    }

    BufferType* const buffer;

private:
    template <typename T, typename ... OtherTypes>
    inline void process(T&& first, OtherTypes&& ... remains) {
        process(std::forward<T>(first));
        process(std::forward<OtherTypes>(remains)...);
    }

    template <typename T>
    inline void process(T&& object){
        buffer->process(std::forward<T>(object));
    }
};


template <typename BufferType>
class OutputBuffer : public Buffer<BufferType> {
    friend class Buffer<BufferType>;
public:
    OutputBuffer(BufferType* const buffer) : Buffer<BufferType>(buffer) {}

    template <typename T>
    inline BufferType& operator<<(T&& arg) {
        process(std::forward<T>(arg));
        return Buffer<BufferType>::getBuffer();
    }
private:
    template <typename T>
    inline void process(T&& object) {
        Serializer<BufferType, typename std::remove_const<typename std::remove_reference<T>::type>::type>::save(*Buffer<BufferType>::buffer, object);
    }
};


class BinaryOutputBuffer : public OutputBuffer<BinaryOutputBuffer> {
public:
    BinaryOutputBuffer(std::ostream& stream)
    : OutputBuffer<BinaryOutputBuffer>(this), m_stream(stream), m_impl_params(nullptr), m_strm(nullptr) {}

    virtual ~BinaryOutputBuffer() = default;

    virtual void write(void const* data, const std::streamsize size) {
        auto const written_size = m_stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        OPENVINO_ASSERT(written_size == size,
                        "[ CPU ] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " +
                            std::to_string(written_size));
    }

    virtual void flush() {}

    void setKernelImplParams(void* impl_params) { m_impl_params = impl_params; }
    void* getKernelImplParams() const { return m_impl_params; }
    void set_stream(void* strm) { m_strm = strm; }
    void* get_stream() const { return m_strm; }

    size_t get_pos() {
        return m_stream.tellp();
    }
private:
    std::ostream& m_stream;
    void* m_impl_params;
    void* m_strm;
};

template <typename BufferType>
class InputBuffer : public Buffer<BufferType> {
    friend class Buffer<BufferType>;
public:
    InputBuffer(BufferType* const buffer) : Buffer<BufferType>(buffer) {}
    // InputBuffer(BufferType* const buffer, dnnl::engine& engine) : Buffer<BufferType>(buffer), m_engine(engine) {}

    template <typename T>
    inline BufferType& operator>>(T&& arg) {
        process(std::forward<T>(arg));
        return Buffer<BufferType>::getBuffer();
    }

    // dnnl::engine& get_engine() { return m_engine; }
private:
    template <typename T>
    inline void process(T&& object) {
        Serializer<BufferType, typename std::remove_reference<T>::type>::load(*Buffer<BufferType>::buffer, object);
    }

    // dnnl::engine& m_engine;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::istream& stream)
    : InputBuffer<BinaryInputBuffer>(this), m_stream(stream), m_impl_params(nullptr) {}
    // BinaryInputBuffer(std::istream& stream, dnnl::engine& engine)
    // : InputBuffer<BinaryInputBuffer>(this, engine), m_stream(stream), m_impl_params(nullptr) {}

    virtual ~BinaryInputBuffer() = default;

    virtual void read(void* const data, std::streamsize size) {
        auto const read_size = m_stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size); // TODO: get raw ptr instead of copy
        OPENVINO_ASSERT(read_size == size,
            "[ CPU ] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

    void setKernelImplParams(void* impl_params) { m_impl_params = impl_params; }
    void* getKernelImplParams() const { return m_impl_params; }

    size_t get_pos() const {
        return m_stream.tellg();
    }

    void* get_ptr() {
        return m_stream.rdbuf().gptr();
    }
private:
    std::istream& m_stream;
    void* m_impl_params;
};

template <typename T>
class Serializer<BinaryOutputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void save(BinaryOutputBuffer& buffer, const T& object) {
printf("-WRITE- T at %llu\n", buffer.get_pos());
        buffer.write(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void load(BinaryInputBuffer& buffer, T& object) {
printf("-READ- T at %llu\n", buffer.get_pos());
        buffer.read(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryOutputBuffer, Data<T>> {
public:
    static void save(BinaryOutputBuffer& buffer, const Data<T>& bin_data) {
printf("-WRITE Data- at %llu\n", buffer.get_pos());
// std::cout << "-WRITE Data-\n";
        buffer.write(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, Data<T>> {
public:
    static void load(BinaryInputBuffer& buffer, Data<T>& bin_data) {
printf("-READ Data- at %llu\n", buffer.get_pos());
// std::cout << "-READ Data-\n";
        buffer.read(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

inline void validate_stream_offset(BinaryInputBuffer& in_buf) {
    const auto act_pos = in_buf.get_pos();
    size_t exp_pos = 0lu;
    in_buf >> exp_pos;

if (exp_pos != act_pos) {
    printf("[ ERROR ] Invalid input stream position. Expected: %llu; Actual: %llu\n",
            exp_pos, act_pos);
}
    OPENVINO_ASSERT(exp_pos == act_pos,
                    "Invalid input stream position. Expected: ",
                    exp_pos,
                    "; Actual: ",
                    act_pos);
}

}  // namespace ov::intel_cpu

#define BIND_BINARY_BUFFER_WITH_TYPE(cls_name)                   \
            namespace ov::intel_cpu {                            \
                BIND_TO_BUFFER(BinaryOutputBuffer, cls_name)     \
                BIND_TO_BUFFER(BinaryInputBuffer, cls_name)      \
            }

// #define BIND_BINARY_BUFFER_WITH_TYPE(cls_name)                   \
//             namespace ov::intel_cpu {                            \
//                 BIND_TO_BUFFER(BinaryOutputBuffer, cls_name)     \
//             }
