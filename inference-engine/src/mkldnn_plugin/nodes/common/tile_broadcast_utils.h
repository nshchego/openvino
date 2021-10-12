// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_parallel.hpp"
//#include "cpu_memcpy.h"
//#include <mkldnn_extension_utils.h>
//#include <ngraph/runtime/host_tensor.hpp>
#include "mkldnn_node.h"

#include <memory>
#include <vector>

class TileBroadcastCommon {
protected:
    static InferenceEngine::SizeVector calculateStridesForDims(const InferenceEngine::SizeVector &dims);
    std::vector<MKLDNNPlugin::NodeDesc> getSupportedConfigs(MKLDNNPlugin::MKLDNNNode *node);
    bool prepareOptimizedParams(MKLDNNPlugin::MKLDNNNode *node, InferenceEngine::SizeVector& srcBlockedDims, InferenceEngine::SizeVector& dstBlockedDims);

    void optimizedExecute(MKLDNNPlugin::MKLDNNNode *node);
//    void ngraphExecute(MKLDNNPlugin::MKLDNNNode *node, std::shared_ptr<ngraph::Node> ngraphNode);

    InferenceEngine::SizeVector repeats;
    bool optimizedCase = false;

private:
    static void fillOptimizedDimsAndSrcStrides(const InferenceEngine::SizeVector &srcBlockedDims, const InferenceEngine::SizeVector &blockedRepeats,
            InferenceEngine::SizeVector &optimizedDims, InferenceEngine::SizeVector &optimizedSrcStrides);

    static bool canBeExecutedInBlockedLayout(const MKLDNNPlugin::VectorDims& srcDims, const InferenceEngine::SizeVector& repeats, size_t elemsInBlock);
    static bool canBeExecutedInNSPCLayout(const MKLDNNPlugin::VectorDims& srcDims, const InferenceEngine::SizeVector& repeats);

    struct {
        std::vector<size_t> dims;
        std::vector<size_t> srcStrides;
        std::vector<size_t> dstStrides;
        size_t copySize;
    } optimizedParams;
};
