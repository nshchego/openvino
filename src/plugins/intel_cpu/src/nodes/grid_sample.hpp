// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <node.h>
#include "kernels/grid_sample_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace ov {
namespace intel_cpu {
namespace node {

class GridSample : public Node {
public:
    GridSample(const std::shared_ptr<ov::Node>& op, const dnnl::engine& eng, WeightsSharing::Ptr &cache);
    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    struct threadExecParams {
//        uint64_t batchNum = 1lu;
        uint64_t channelsNum = 1lu;
        float srcWidthF;
        float srcHeightF;
        uint64_t srcWidthB = 1lu;
        uint64_t dstStartB = 0lu;
        uint64_t gridStartB = 0lu;
//        uint64_t srcBatchStepB = 0lu;
        uint64_t srcChannelStepB = 0lu;
        uint64_t dstChannelStepB = 0lu;
        uint64_t dstBatchStepB = 0lu;
        uint64_t gridBatchStepB = 0lu;
        float srcHeightMul2F = 1.f;
        float srcWidthMul2F = 1.f;
        float srcHeightMul2Sub1F = 1.f;
        float srcWidthMul2Sub1F = 1.f;
        float srcHeightSub1F = 0.f;
        float srcWidthSub1F = 0.f;
        float wDenormCoef = 1.f;
        float hDenormCoef = 1.f;
        const float halfVal = 0.5f;
        const float one = 1.f;
        uint64_t workAmount = 0lu;
    };

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
//    bool needPrepareParams() const override;
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    bool alignCorners = false;
    InterpolationMode interpolationMode = InterpolationMode::BILINEAR;
    PaddingMode paddingMode = PaddingMode::ZEROS;

    uint64_t dataTypeSize = 1lu;
    uint64_t gridTypeSize = 1lu;

    std::vector<threadExecParams> execParamsPerThread;

    static constexpr size_t IN_DATA = 0;
    static constexpr size_t IN_GRID = 1;

    std::shared_ptr<jitGridSampleKernelBase> jitKernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
