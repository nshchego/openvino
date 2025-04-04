// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ostream>


#include "helpers.hpp"
#include "onednn/dnnl.h"
#include "openvino/core/except.hpp"
#include "serializer.hpp"

namespace ov {
namespace intel_cpu {

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
    : OutputBuffer<BinaryOutputBuffer>(this), stream(stream), _impl_params(nullptr), _strm(nullptr) {}

    virtual ~BinaryOutputBuffer() = default;

    virtual void write(void const* data, const std::streamsize size) {
        auto const written_size = stream.rdbuf()->sputn(reinterpret_cast<const char*>(data), size);
        OPENVINO_ASSERT(written_size == size,
                        "[GPU] Failed to write " + std::to_string(size) + " bytes to stream! Wrote " +
                            std::to_string(written_size));
    }

    virtual void flush() {}

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }
    void set_stream(void* strm) { _strm = strm; }
    void* get_stream() const { return _strm; }

private:
    std::ostream& stream;
    void* _impl_params;
    void* _strm;
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
    : InputBuffer<BinaryInputBuffer>(this), _stream(stream), _impl_params(nullptr) {}
    // BinaryInputBuffer(std::istream& stream, dnnl::engine& engine)
    // : InputBuffer<BinaryInputBuffer>(this, engine), _stream(stream), _impl_params(nullptr) {}

    virtual ~BinaryInputBuffer() = default;

    virtual void read(void* const data, std::streamsize size) {
        auto const read_size = _stream.rdbuf()->sgetn(reinterpret_cast<char*>(data), size);
        OPENVINO_ASSERT(read_size == size,
            "[GPU] Failed to read " + std::to_string(size) + " bytes from stream! Read " + std::to_string(read_size));
    }

    void setKernelImplParams(void* impl_params) { _impl_params = impl_params; }
    void* getKernelImplParams() const { return _impl_params; }

private:
    std::istream& _stream;
    void* _impl_params;
};

template <typename T>
class Serializer<BinaryOutputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void save(BinaryOutputBuffer& buffer, const T& object) {
printf("-WRITE T-\n");
        buffer.write(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, T, typename std::enable_if<std::is_arithmetic<T>::value>::type> {
public:
    static void load(BinaryInputBuffer& buffer, T& object) {
printf("-READ T-\n");
        buffer.read(std::addressof(object), sizeof(object));
    }
};

template <typename T>
class Serializer<BinaryOutputBuffer, Data<T>> {
public:
    static void save(BinaryOutputBuffer& buffer, const Data<T>& bin_data) {
// printf("-WRITE Data-\n");
std::cout << "-WRITE Data-\n";
        buffer.write(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

template <typename T>
class Serializer<BinaryInputBuffer, Data<T>> {
public:
    static void load(BinaryInputBuffer& buffer, Data<T>& bin_data) {
// printf("-READ Data-\n");
std::cout << "-READ Data-\n";
        buffer.read(bin_data.data, static_cast<std::streamsize>(bin_data.number_of_bytes));
    }
};

}  // namespace intel_cpu
}  // namespace ov
