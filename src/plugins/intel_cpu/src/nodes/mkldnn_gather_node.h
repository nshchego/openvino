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
    void execute(mkldnn::stream strm) override;
    bool created() const override;

    static bool isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept;

protected:
    void executeDynamicImpl(mkldnn::stream strm) override;
    bool needPrepareParams() const override;
    void prepareParams() override;

private:
    //inline void initParams();
    void initShortParams(std::vector<int>& shortPermIdx, std::vector<int>& shortBeforeAxisDiff, uint64_t start);
    void execReference();

    int axis = 0;
    int batchDims = 0;
    bool reverseIndexing = false;
    size_t dataTypeSize = 1lu;
    int dataSrcRank = 1;
    bool isAxisInputConst = false;
    std::string errorPrefix;
    static constexpr uint64_t idxTypeSize = sizeof(int);

    int axisDim;
    uint64_t beforeBatchSize;
    uint64_t betweenBatchAndAxisSize;
    uint64_t afterAxisSize;
    uint64_t specIndicesSize;
    uint64_t afterAxisSizeInBytes;
    uint64_t axisAndAfterAxisSizeInBytes;
    uint64_t srcAfterBatchSizeInBytes;
    uint64_t totalWork;

    std::vector<std::vector<int>> shortPermIdx;
    std::vector<std::vector<int>> shortBeforeAxisDiff;
    std::vector<std::vector<int>> specIndicesInBytes;
    std::vector<std::vector<int>> idxBatchSumInBytes;
    std::vector<std::vector<int>> dataBeforeAxisSumBPerTr;
    std::vector<int> betweenBatchAndAxisIters;

    static constexpr size_t GATHER_DATA = 0;
    static constexpr size_t GATHER_INDICES = 1;
    static constexpr size_t GATHER_AXIS = 2;

    std::shared_ptr<jitGatherKernelBase> jitKernel;
};

}  // namespace MKLDNNPlugin
