// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <mkldnn_node.h>
#include "kernels/gather_uni_kernel.hpp"

#include <memory>
#include <string>
#include <vector>

namespace MKLDNNPlugin {

class MKLDNNGatherNode : public MKLDNNNode {
public:
    MKLDNNGatherNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache);

    void getSupportedDescriptors() override {};
    void initSupportedPrimitiveDescriptors() override;
    void createPrimitive() override;
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(mkldnn::stream strm) override;
    bool needPrepareParams() const override;
    void prepareParams() override;
    std::vector<VectorDims> shapeInfer() const override;

private:
    void initShortParams(uint64_t ithr, uint64_t start);
    void execReference();

    int axis = 0;
    int batchDims = 0;
    bool reverseIndexing = false;
    int dataSrcRank = 1;
    bool isDataShapeStat = false;
    bool isIdxShapeStat = false;
    bool isAxisInputConst = false;
    uint64_t dataTypeSize = 1lu;
    static constexpr uint64_t idxTypeSize = sizeof(int);

    int axisDim;
    uint64_t beforeBatchSize;
    uint64_t beforeAxisSize;
    uint64_t betweenBatchAndAxisSize;
    uint64_t afterAxisSize;
    uint64_t specIndicesSize;
    uint64_t afterAxisSizeInBytes;
    uint64_t axisAndAfterAxisSizeInBytes;
    uint64_t srcAfterBatchSizeInBytes;
    uint64_t specIdxAndAfterAxSizeB;
    uint64_t totalWork;

    std::vector<std::vector<int>> specIdxInBytesPerThr;
    std::vector<std::vector<int>> permIdxMaskPerThr;
    std::vector<std::vector<int>> srcBeforeAxisDiffPerThr;
    std::vector<std::vector<int>> idxBatchSumInBytes;
    std::vector<std::vector<int>> dataBeforeAxisSumInBytesPerThr;

    std::vector<std::vector<int>> afterAxIdxInBytesPerThr;
    std::vector<std::vector<int>> specIdxDiffPerThr;
    std::vector<std::vector<int>> beforeAxPermMaskPerThr;
    std::vector<std::vector<int>> afterAxPermMaskPerThr;
    std::vector<int> betweenBatchAndAxisIters;
    std::vector<int> specIdxAndAfterAxIterBPerThr;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;

    std::shared_ptr<jitGatherKernelBase> jitKernel;
};

}  // namespace MKLDNNPlugin
