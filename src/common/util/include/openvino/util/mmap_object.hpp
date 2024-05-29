// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for definition of abstraction over platform specific shared memory map objects
 * @file mmap_object.hpp
 */

#pragma once

#include <memory>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <string>

namespace ov {

/**
 * @brief This class represents a mapped memory.
 * Instead of reading files, we can map the memory via mmap for Linux or MapViewOfFile for Windows.
 * The MappedMemory class is a abstraction to handle such memory with os-dependent details.
 */
class MappedMemory {
public:
    virtual ~MappedMemory() = default;

    virtual char* data() noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual size_t get_offset() const = 0;
    virtual void set_offset(size_t offset) = 0;
};

/**
 * @brief Returns mapped memory for a file from provided path.
 * Instead of reading files, we can map the memory via mmap for Linux
 * in order to avoid time-consuming reading and reduce memory consumption.
 *
 * @param path Path to a file which memory will be mmaped.
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::string& path);

#ifdef OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

/**
 * @brief Returns mapped memory for a file from provided path.
 * Instead of reading files, we can map the memory via MapViewOfFile for Windows
 * in order to avoid time-consuming reading and reduce memory consumption.
 *
 * @param path Path to a file which memory will be mmaped.
 * @return MappedMemory shared ptr object which keep mmaped memory and control the lifetime.
 */
std::shared_ptr<ov::MappedMemory> load_mmap_object(const std::wstring& path);

#endif  // OPENVINO_ENABLE_UNICODE_PATH_SUPPORT

class mmap_stream: public std::istream {
public:
    // mmap_stream(const std::shared_ptr<ov::MappedMemory>& mm) {
    mmap_stream(const std::shared_ptr<ov::MappedMemory>& mm, std::filebuf* str = nullptr) : std::istream(str) {
        m_mapped_memory = mm;
    }
    mmap_stream() = default;
    // explicit mmap_stream(FILE* cstream) {

    // };

private:
    std::shared_ptr<ov::MappedMemory> m_mapped_memory;
};

// class mmap_stream: public std::ios_base {
// public:
//     mmap_stream(const std::shared_ptr<ov::MappedMemory>& mm) {
//         m_mapped_memory = mm;
//     }
//     // mmap_stream() = default;
//     // explicit mmap_stream(FILE* cstream) {

//     // };

// private:
//     std::shared_ptr<ov::MappedMemory> m_mapped_memory;
// };

}  // namespace ov
