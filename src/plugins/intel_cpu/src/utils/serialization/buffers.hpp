// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <iostream>

#include "helpers.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "serializer.hpp"

namespace ov::intel_cpu {

template <typename BufferType>
class Buffer {
public:
    Buffer(BufferType* const buffer) : buffer(buffer) {}

    virtual ~Buffer() {}

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
    BinaryOutputBuffer(std::ostream& stream);

    virtual void write(const void* data, const std::streamsize size) {
        auto const written_size = m_stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
if (written_size != size) {  // TODO: remove
    std::cout << "BinaryOutputBuffer::write\n";
}
        OPENVINO_ASSERT(written_size == size,
                        "[ CPU ] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " +
                            std::to_string(written_size));
    }

    virtual void flush() {}

    // void set_kernel_impl_params(void* impl_params) { m_impl_params = impl_params; }

    // void* get_kernel_impl_params() const { return m_impl_params; }

    // void set_stream(void* strm) { m_strm = strm; }

    // void* get_stream() const { return m_strm; }

    size_t get_position() {
        return m_stream.tellp();
    }

    void dump_position();

private:
    std::ostream& m_stream;
    void* m_impl_params;
    void* m_strm;  // TODO: remove?
    size_t m_header_offset;
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
    // SrcType& m_source;
};

class BinaryInputBuffer : public InputBuffer<BinaryInputBuffer> {
public:
    BinaryInputBuffer(std::streampos header_offset = 0);

    // BinaryInputBuffer(SrcType& stream);

    // BinaryInputBuffer(std::istream& stream);

    // BinaryInputBuffer(ov::Tensor& stream);

    // BinaryInputBuffer(std::istream& stream, dnnl::engine& engine)
    // : InputBuffer<BinaryInputBuffer>(this, engine), m_stream(stream), m_impl_params(nullptr) {}

    virtual void read(void* const data, std::streamsize size);

    virtual void read(const void*& data, std::streamsize size);

    // void set_kernel_impl_params(void* impl_params) { m_impl_params = impl_params; }

    // void* get_kernel_impl_params() const { return m_impl_params; }

    virtual size_t get_position() const = 0;

    virtual void check_position();

    // virtual const std::streambuf* rdbuf() = 0;
    virtual const void* get_data() = 0;

    // virtual std::istream& seekg(std::istream::off_type offset, std::ios_base::seekdir way) = 0;
    virtual void seekg(std::istream::off_type offset, std::ios_base::seekdir way = std::ios_base::cur) = 0;

protected:
    std::streampos m_header_offset;

private:
    // std::istream& m_stream;
    // void* m_impl_params;
};

class StreamInputBuffer : public BinaryInputBuffer {
public:
    StreamInputBuffer(std::istream& stream, std::streampos header_offset = 0);

    void read(void* const data, std::streamsize size) override;

    size_t get_position() const override;

    // void check_position();

    // const std::streambuf* rdbuf() override;
    const void* get_data();

    // std::istream& seekg(std::istream::off_type offset, std::ios_base::seekdir way);
    void seekg(std::istream::off_type offset, std::ios_base::seekdir way = std::ios_base::cur) override;

private:
    std::istream& m_stream;
};

class TensorInputBuffer : public BinaryInputBuffer {
public:
    TensorInputBuffer(const ov::Tensor& model_tensor, std::streampos header_offset = 0);

    void read(void* const data, std::streamsize size) override;

    void read(const void*& data, std::streamsize size) override;

    size_t get_position() const override;

    // void check_position();

    // const std::streambuf* rdbuf() override;
    const void* get_data();

    // std::istream& seekg(std::istream::off_type offset, std::ios_base::seekdir way);
    void seekg(std::istream::off_type offset, std::ios_base::seekdir way = std::ios_base::cur) override;

private:
    const ov::Tensor& m_model_tensor;
    const uint8_t* m_data = nullptr;
    size_t m_offset = 0UL;
};

template <typename T>
class Serializer<BinaryOutputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void save(BinaryOutputBuffer& buffer, const T& object) {
// printf("-WRITE- T at %llu\n", buffer.get_position());
        buffer.write(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void load(BinaryInputBuffer& buffer, T& object) {
// printf("-READ- T at %llu\n", buffer.get_position());
        buffer.read(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryOutputBuffer, Data<T>> {
public:
    static void save(BinaryOutputBuffer& buffer, const Data<T>& bin_data) {
// printf("-WRITE Data- at %llu\n", buffer.get_position());
        buffer.write(bin_data.m_data, static_cast<std::streamsize>(bin_data.m_number_of_bytes));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, Data<T>> {
public:
    static void load(BinaryInputBuffer& buffer, Data<T>& bin_data) {
// printf("-READ Data- at %llu\n", buffer.get_position());
        buffer.read(bin_data.m_data, static_cast<std::streamsize>(bin_data.m_number_of_bytes));
    }
};

// inline void validate_stream_offset(BinaryInputBuffer& in_buf) {
//     const auto act_pos = in_buf.get_position();
//     size_t exp_pos = 0lu;
//     in_buf >> exp_pos;

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
