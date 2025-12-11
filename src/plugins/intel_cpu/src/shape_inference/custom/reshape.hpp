// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <node.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/node.hpp"
#include "shape_inference/shape_inference_cpu.hpp"

#pragma once

namespace ov::intel_cpu::node {
using Result = IShapeInfer::Result;
class ReshapeShapeInfer : public ShapeInferEmptyPads {
public:
    ReshapeShapeInfer() = default;

    explicit ReshapeShapeInfer(bool specialZero) : m_specialZero(specialZero) {}
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

    DECLARE_SERIALIZATION_OBJECT_MEMBERS_OVERRIDE(ov::intel_cpu::node::ReshapeShapeInfer)

    void save(BinaryOutputBuffer& out_buf) const override;

    void load(BinaryInputBuffer& in_buf) override;

private:
    bool m_specialZero;
};

class SqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    SqueezeShapeInfer() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

    DECLARE_SERIALIZATION_OBJECT_MEMBERS_OVERRIDE(ov::intel_cpu::node::SqueezeShapeInfer)
};

class UnsqueezeShapeInfer : public ShapeInferEmptyPads {
public:
    UnsqueezeShapeInfer() = default;
    Result infer(const std::vector<std::reference_wrapper<const VectorDims>>& input_shapes,
                 const std::unordered_map<size_t, MemoryPtr>& data_dependency) override;
    [[nodiscard]] port_mask_t get_port_mask() const override {
        return PortMask(1);
    }

    DECLARE_SERIALIZATION_OBJECT_MEMBERS_OVERRIDE(ov::intel_cpu::node::UnsqueezeShapeInfer)
};

class ReshapeShapeInferFactory : public ShapeInferFactory {
public:
    explicit ReshapeShapeInferFactory(std::shared_ptr<ov::Node> op) : m_op(std::move(op)) {}
    [[nodiscard]] ShapeInferPtr makeShapeInfer() const override;

private:
    std::shared_ptr<ov::Node> m_op;
};
}  // namespace ov::intel_cpu::node
