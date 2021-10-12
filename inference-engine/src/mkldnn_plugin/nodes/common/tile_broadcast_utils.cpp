// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tile_broadcast_utils.h"
#include "cpu_memcpy.h"
#include <memory_desc/cpu_memory_desc_utils.h>
#include "memory_desc/dnnl_blocked_memory_desc.h"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

SizeVector TileBroadcastCommon::calculateStridesForDims(const SizeVector &dims) {
    SizeVector strides(dims.size(), 1);

    for (int i = strides.size() - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * dims[i + 1];
    }

    return strides;
}

void TileBroadcastCommon::fillOptimizedDimsAndSrcStrides(const SizeVector &srcBlockedDims, const SizeVector &blockedRepeats,
        SizeVector &optimizedDims, SizeVector &optimizedSrcStrides) {
    SizeVector srcBlockedStrides = calculateStridesForDims(srcBlockedDims);

    for (int i = 0; i < srcBlockedDims.size(); i++) {
        optimizedDims.push_back(blockedRepeats[i]);
        optimizedDims.push_back(srcBlockedDims[i]);
        optimizedSrcStrides.push_back(0);
        optimizedSrcStrides.push_back(srcBlockedStrides[i]);
    }

    int i = 1;
    while (i < optimizedDims.size() - 1) {
        if (optimizedDims[i] == 1) {
            optimizedDims[i + 1] *= optimizedDims[i - 1];
            optimizedDims.erase(optimizedDims.begin() + i - 1, optimizedDims.begin() + i + 1);
            optimizedSrcStrides.erase(optimizedSrcStrides.begin() + i - 1, optimizedSrcStrides.begin() + i + 1);
        } else {
            i++;
        }
    }

    if (optimizedDims[0] == 1 && optimizedDims.size() > 1) {
        optimizedDims.erase(optimizedDims.begin());
        optimizedSrcStrides.erase(optimizedSrcStrides.begin());
    }

    if (optimizedDims[optimizedDims.size() - 1] == 1 && optimizedDims.size() > 1) {
        optimizedDims.erase(optimizedDims.end() - 1);
        optimizedSrcStrides.erase(optimizedSrcStrides.end() - 1);
    }
}

bool TileBroadcastCommon::canBeExecutedInBlockedLayout(const VectorDims& srcDims, const SizeVector& repeats,
        size_t elemsInBlock) {
    if (repeats[1] != 1 && srcDims[1] % elemsInBlock != 0)
        return false;

    SizeVector srcBlockedDims = srcDims;
    SizeVector blockedRepeats = repeats;
    srcBlockedDims[1] = (srcBlockedDims[1] + elemsInBlock - 1) / elemsInBlock;
    srcBlockedDims.push_back(elemsInBlock);
    blockedRepeats.push_back(1);

    SizeVector optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    const int maxNDims = 6;
    return optimizedDims.size() <= maxNDims;
}

bool TileBroadcastCommon::canBeExecutedInNSPCLayout(const VectorDims& srcDims, const SizeVector &repeats) {
    SizeVector srcBlockedDims = srcDims;
    SizeVector blockedRepeats = repeats;
    srcBlockedDims.push_back(srcBlockedDims[1]);
    srcBlockedDims.erase(srcBlockedDims.begin() + 1);
    blockedRepeats.push_back(blockedRepeats[1]);
    blockedRepeats.erase(blockedRepeats.begin() + 1);

    SizeVector optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    const int maxNDims = 6;
    return optimizedDims.size() <= maxNDims;
}

std::vector<NodeDesc> TileBroadcastCommon::getSupportedConfigs(MKLDNNNode *node) {
std::cout << "TileBroadcastCommon::getSupportedConfigs" << std::endl;
    std::vector<NodeDesc> supportedPrimitiveDescriptors;
    auto precision = node->getOriginalInputPrecisionAtPort(0);
    auto dataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);

    const auto& srcDims = node->getInputShapeAtPort(0).getDims();
    const auto& dstDims = node->getOutputShapeAtPort(0).getDims();

    NodeConfig config;
    if (repeats.size() != dstDims.size())
        IE_THROW() << node->getTypeStr() << " node with name " << node->getName() << " has incorrect Repeats vector."
                "Repeats size must be equal to dstDims size. Repeats size: " << repeats.size() << ", dstDims size: " << dstDims.size();

    config.dynBatchSupport = false;
    config.inConfs.resize(node->getParentEdges().size());
    config.inConfs[0].inPlace = -1;
    config.inConfs[0].constant = false;
    config.inConfs[1].inPlace = -1;
    config.inConfs[1].constant = true;
    config.inConfs[1].desc = std::make_shared<CpuBlockedMemoryDesc>(Precision::I32, node->getInputShapeAtPort(1));
    if (config.inConfs.size() == 3) {
        config.inConfs[2].inPlace = -1;
        config.inConfs[2].constant = true;
        config.inConfs[2].desc = std::make_shared<CpuBlockedMemoryDesc>(Precision::I32, node->getInputShapeAtPort(2));
    }

    config.outConfs.resize(node->getChildEdges().size());

    auto pushDesc = [&](mkldnn::memory::format_tag inFormat, mkldnn::memory::format_tag outFormat) {
        config.inConfs[0].desc = std::make_shared<DnnlBlockedMemoryDesc>(node->getInputShapeAtPort(0), dataType, inFormat);
        for (int i = 0; i < config.outConfs.size(); i++) {
            config.outConfs[i].inPlace = -1;
            config.outConfs[i].constant = false;
            config.outConfs[i].desc = std::make_shared<DnnlBlockedMemoryDesc>(node->getOutputShapeAtPort(0), dataType, outFormat);
        }
        supportedPrimitiveDescriptors.push_back({config, impl_desc_type::ref});
    };

    if (srcDims.size() == dstDims.size() && (dstDims.size() == 4 || dstDims.size() == 5)) {
        if (canBeExecutedInBlockedLayout(srcDims, repeats, 16)) {
            if (dstDims.size() == 4) {
                pushDesc(mkldnn::memory::format_tag::nChw16c, mkldnn::memory::format_tag::nChw16c);
            } else {
                pushDesc(mkldnn::memory::format_tag::nCdhw16c, mkldnn::memory::format_tag::nCdhw16c);
            }
        }
        if (canBeExecutedInBlockedLayout(srcDims, repeats, 8)) {
            if (dstDims.size() == 4) {
                pushDesc(mkldnn::memory::format_tag::nChw8c, mkldnn::memory::format_tag::nChw8c);
            } else {
                pushDesc(mkldnn::memory::format_tag::nCdhw8c, mkldnn::memory::format_tag::nCdhw8c);
            }
        }
        if (canBeExecutedInNSPCLayout(srcDims, repeats)) {
            if (dstDims.size() == 4) {
                pushDesc(mkldnn::memory::format_tag::nhwc, mkldnn::memory::format_tag::nhwc);
            } else {
                pushDesc(mkldnn::memory::format_tag::ndhwc, mkldnn::memory::format_tag::ndhwc);
            }
        }
    }

    pushDesc(MKLDNNExtensionUtils::GetPlainFormatByRank(srcDims.size()), MKLDNNExtensionUtils::GetPlainFormatByRank(dstDims.size()));

    return supportedPrimitiveDescriptors;
}

bool TileBroadcastCommon::prepareOptimizedParams(MKLDNNNode *node, SizeVector& srcBlockedDims, SizeVector& dstBlockedDims) {
    while (srcBlockedDims.size() < dstBlockedDims.size()) {
        srcBlockedDims.insert(srcBlockedDims.begin(), 1);
    }

    SizeVector blockedRepeats = repeats;
    // for nC(d)hw16c and nC(d)hw8c layouts
    while (blockedRepeats.size() < dstBlockedDims.size()) {
        blockedRepeats.push_back(1);
    }
    // for NSPC layouts
    if (node->getBaseMemDescAtInputPort(0)->hasLayoutType(LayoutType::nspc) || one_of(node->getBaseMemDescAtInputPort(0)->getShape().getRank(), 4, 5)) {
        blockedRepeats.push_back(blockedRepeats[1]);
        blockedRepeats.erase(blockedRepeats.begin() + 1);
    }

    SizeVector optimizedDims, optimizedSrcStrides;
    fillOptimizedDimsAndSrcStrides(srcBlockedDims, blockedRepeats, optimizedDims, optimizedSrcStrides);

    const int maxNDims = 6;
    if (optimizedDims.size() > maxNDims)
        return false;

    while (optimizedDims.size() < maxNDims) {
        optimizedDims.insert(optimizedDims.begin(), 1);
        optimizedSrcStrides.insert(optimizedSrcStrides.begin(), 1);
    }

    SizeVector optimizedDstStrides = calculateStridesForDims(optimizedDims);

    size_t dataSize = node->getSelectedPrimitiveDescriptor()->getConfig().inConfs[0].desc->getPrecision().size();
    for (int i = 0; i < optimizedDims.size(); i++) {
        optimizedSrcStrides[i] *= dataSize;
        optimizedDstStrides[i] *= dataSize;
    }

    optimizedParams.dims = optimizedDims;
    optimizedParams.srcStrides = optimizedSrcStrides;
    optimizedParams.dstStrides = optimizedDstStrides;
    optimizedParams.copySize = optimizedDims[5] * dataSize;

    return true;
}

void TileBroadcastCommon::optimizedExecute(MKLDNNNode *node) {
    auto srcData = reinterpret_cast<const char *>(node->getParentEdgeAt(0)->getMemory().GetPtr());
    auto dstData = reinterpret_cast<char *>(node->getChildEdgeAt(0)->getMemory().GetPtr());

    if (optimizedParams.srcStrides[5] == 0) {
        parallel_for5d(optimizedParams.dims[0], optimizedParams.dims[1], optimizedParams.dims[2], optimizedParams.dims[3], optimizedParams.dims[4],
                [&](int i0, int i1, int i2, int i3, int i4) {
            auto srcData2 = srcData + (i0 * optimizedParams.srcStrides[0] + i1 * optimizedParams.srcStrides[1] +
                                                 i2 * optimizedParams.srcStrides[2] + i3 * optimizedParams.srcStrides[3] +
                                                 i4 * optimizedParams.srcStrides[4]);
            auto dstData2 = dstData + (i0 * optimizedParams.dstStrides[0] + i1 * optimizedParams.dstStrides[1] +
                                           i2 * optimizedParams.dstStrides[2] + i3 * optimizedParams.dstStrides[3] +
                                           i4 * optimizedParams.dstStrides[4]);
            for (int i = 0; i < optimizedParams.dims[5]; i++) {
                cpu_memcpy(dstData2 + i * optimizedParams.dstStrides[5], srcData2, optimizedParams.dstStrides[5]);
            }
        });
    } else {
        parallel_for5d(optimizedParams.dims[0], optimizedParams.dims[1], optimizedParams.dims[2], optimizedParams.dims[3], optimizedParams.dims[4],
                [&](int i0, int i1, int i2, int i3, int i4) {
            auto srcData2 = srcData + (i0 * optimizedParams.srcStrides[0] + i1 * optimizedParams.srcStrides[1] +
                                                 i2 * optimizedParams.srcStrides[2] + i3 * optimizedParams.srcStrides[3] +
                                                 i4 * optimizedParams.srcStrides[4]);
            auto dstData2 = dstData + (i0 * optimizedParams.dstStrides[0] + i1 * optimizedParams.dstStrides[1] +
                                           i2 * optimizedParams.dstStrides[2] + i3 * optimizedParams.dstStrides[3] +
                                           i4 * optimizedParams.dstStrides[4]);
            cpu_memcpy(dstData2, srcData2, optimizedParams.copySize);
        });
    }
}

//void TileBroadcastCommon::ngraphExecute(MKLDNNNode *node, std::shared_ptr<ov::Node> ngraphNode) {
//    ngraph::HostTensorVector inputs, outputs;
//    auto srcDataPtr = node->getParentEdgeAt(0)->getMemory().GetPtr();
//    inputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphNode->input(0).get_element_type(), ngraphNode->input(0).get_shape(), srcDataPtr));
//    std::vector<int64_t> newData;
//    // WA: for Tile Node we need cast data to int64_t data type
//    if (std::string(ngraphNode->get_type_name()) == "Tile") {
//        auto dataPtr = node->getParentEdgeAt(1)->getMemory().GetPtr();
//        newData.resize(node->getParentEdgeAt(1)->getMemory().GetShape().getElementsCount());
//        for (int j = 0; j < node->getParentEdgeAt(1)->getMemory().GetShape().getElementsCount(); j++) {
//            newData[j] = reinterpret_cast<const int32_t*>(dataPtr)[j];
//        }
//        inputs.push_back(std::make_shared<ngraph::HostTensor>(ngraph::element::i64, ngraphNode->input(1).get_shape(), newData.data()));
//    } else {
//        for (int i = 1; i < node->getParentEdges().size(); i++) {
//            auto dataPtr = node->getParentEdgeAt(i)->getMemory().GetPtr();
//            inputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphNode->input(i).get_element_type(), ngraphNode->input(i).get_shape(), dataPtr));
//        }
//    }
//
//    for (int i = 0; i < node->getChildEdges().size(); i++) {
//        auto dataPtr = node->getChildEdgeAt(i)->getMemory().GetPtr();
//        outputs.push_back(std::make_shared<ngraph::HostTensor>(ngraphNode->output(i).get_element_type(), ngraphNode->output(i).get_shape(), dataPtr));
//    }
//    ngraphNode->evaluate(outputs, inputs);
//}
