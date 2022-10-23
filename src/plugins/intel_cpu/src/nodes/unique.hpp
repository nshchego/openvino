// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/unique.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class Unique : public Node {
public:
    Unique(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    void flattenTensorExec();
    void slicedTensorExec();

    bool sorted = false;
    int axis = -1;
    bool definedOutputs[4] = { false, false, false, false };
    uint64_t dataTypeSize = 1lu;
    InferenceEngine::Precision dataPrecision;

    int threadsNum = 1;
    std::vector<UniqueKernelExecArgs> execParamsPerThread;

    static constexpr size_t IN_DATA = 0;
    static constexpr size_t AXIS    = 1;

    std::shared_ptr<UniqueKernelBase> kernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
