// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "serialize_base.hpp"

#include "openvino/runtime/aligned_buffer.hpp"
#include "openvino/util/mmap_object.hpp"

namespace ov {
namespace intel_cpu {

class ModelMmapDeserializer : public ModelDeserializerBase {
public:
    typedef std::function<std::shared_ptr<ov::Model>(const std::shared_ptr<ov::AlignedBuffer>&, const std::shared_ptr<ov::AlignedBuffer>&)> model_builder;

    ModelMmapDeserializer(const std::shared_ptr<ov::MappedMemory>& buffer, model_builder fn);

    void parse(std::shared_ptr<ov::Model>& model) override;

private:
    std::shared_ptr<ov::MappedMemory> m_model_buffer;
    model_builder m_model_builder;
};

}   // namespace intel_cpu
}   // namespace ov
