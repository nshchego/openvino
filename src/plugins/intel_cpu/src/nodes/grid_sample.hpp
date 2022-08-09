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

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(dnnl::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept;

    struct threadExecParams {
        uint64_t batchNum = 1lu;
        uint64_t channelsNum = 1lu;
        float srcWidthFl;
        float srcHeightFl;
        uint64_t workAmount = 0lu;
        uint64_t dstStartB = 0lu;
        uint64_t gridStartB = 0lu;
        uint64_t srcBatchStepB = 0lu;
        uint64_t srcChannelStepB = 0lu;
        uint64_t dstChannelStepB = 0lu;
        uint64_t dstBatchStepB = 0lu;
        uint64_t gridBatchStepB = 0lu;
        float wDenormCoef = 1.f;
        float hDenormCoef = 1.f;
    };

protected:
    void executeDynamicImpl(dnnl::stream strm) override;
//    bool needPrepareParams() const override;
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer() const override;

private:
//    void initShortParams(threadExecParams& p, uint64_t start);
//    void execReference();

    bool alignCorners = false;
    InterpolationMode interpolationMode = InterpolationMode::BILINEAR;
    PaddingMode paddingMode = PaddingMode::ZEROS;

//    bool isDataShapeStat = false;
//    bool isIdxShapeStat = false;
//    bool isAxisInputConst = false;
//
//    bool reverseIndexing = false;
//
    uint64_t dataTypeSize = 1lu;
    uint64_t gridTypeSize = 1lu;
//    static constexpr uint64_t idxTypeSize = sizeof(int);
//
//    int axis = 0;
//    int axisDim = 0;
//    int batchDims = 0;
//    int dataSrcRank = 1;
//    uint64_t specIndicesSize = 0lu;
//    uint64_t beforeBatchSize = 0lu;
//    uint64_t beforeAxisSize = 0lu;
//    uint64_t betweenBatchAndAxisSize = 0lu;
//    uint64_t afterAxisSize = 0lu;
//    uint64_t afterAxisSizeInBytes = 0lu;
//    uint64_t axisAndAfterAxisSizeInBytes = 0lu;
//    uint64_t srcAfterBatchSizeInBytes = 0lu;
//    uint64_t specIdxAndAfterAxSizeB = 0lu;
    uint64_t totalWork = 0lu;
//
    std::vector<threadExecParams> execParamsPerThread;

    static constexpr size_t IN_DATA = 0;
    static constexpr size_t IN_GRID = 1;

    std::shared_ptr<jitGridSampleKernelBase> jitKernel;
};

}   // namespace node
}   // namespace intel_cpu
}   // namespace ov
