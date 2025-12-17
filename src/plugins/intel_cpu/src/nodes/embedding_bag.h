// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

#include "cpu_memory.h"
#include "cpu_types.h"
#include "openvino/core/node.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov::intel_cpu::node {

class EmbeddingBag {
public:
    enum class Reduction : uint8_t { SUM, MEAN };
    EmbeddingBag(const std::shared_ptr<ov::Node>& op,
                 size_t requiredInputNum,
                 size_t indicesIdx,
                 size_t perSampleWeightsIdx,
                 size_t defaultIndexIdx);

    EmbeddingBag(BinaryInputBuffer& in_buf);

    void execute(const uint8_t* srcData,
                 const uint8_t* weightsData,
                 const ov::element::Type& srcPrc,
                 const VectorDims& inDims,
                 const MemoryPtr& outMemory);

    virtual ~EmbeddingBag() = default;

    void save(BinaryOutputBuffer& out_buf) const;

    void load(BinaryInputBuffer& in_buf);

protected:
    virtual void initFromInputs() = 0;
    virtual void getIndices(size_t embIndex,
                            const int*& indicesRef,
                            size_t& size,
                            int& weightsIdx,
                            bool& withWeights) = 0;

    void prepareParams(const VectorDims& indexStaticShape);

    template <typename T>
    void processData(const T* srcData, const T* weightsData, const VectorDims& inDataDims, const MemoryPtr& outMemory);

    const size_t EMB_TABLE_IDX = 0LU;
    size_t m_indices_index;
    size_t m_per_sample_weights_index;
    size_t m_default_index;

    Reduction _reduction = Reduction::SUM;
    bool _withWeights = false;
    size_t _embDepth = 0;
    std::string _layerName;
};

}  // namespace ov::intel_cpu::node
