// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_tile_node.h"
#include "common/cpu_memcpy.h"

using namespace InferenceEngine;
using namespace MKLDNNPlugin;

bool MKLDNNTileNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v0::Tile::type_info)) {
            errorMessage = "Only opset1 Tile operation is supported";
            return false;
        }
//        if (op->get_input_shape(TILE_INPUT).size() != op->get_input_shape(TILE_REPEATS)[0]) {
//            errorMessage = "Doesn't support inputs with different ranks";
//            return false;
//        }
        if (!isDynamicNgraphNode(op) &&
                op->get_input_node_ptr(TILE_REPEATS)->get_type_info() != ov::op::v0::Constant::type_info) {
            errorMessage = "Only const 'repeats' input is supported";
            return false;
        }
//        const auto repeats = repeatsNode->cast_vector<int32_t>();
//        if (std::count_if(repeats.begin(), repeats.end(), [](int32_t x) { return x > 1; }) > 1) {
//            errorMessage = "Doesn't support 'repeats' with more than one specified axis";
//            return false;
//        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNTileNode::MKLDNNTileNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Tile node with name '" + getName() + "'";

    if (op->get_input_node_ptr(TILE_REPEATS)->get_type_info() == ov::op::v0::Constant::type_info) {
        constMap[TILE_INPUT] = true;
        repeats = ov::as_type<const ov::op::v0::Constant>(op->get_input_node_ptr(TILE_REPEATS))->cast_vector<uint64_t>();
        while (repeats.size() < getInputShapeAtPort(TILE_INPUT).getRank()) {
            repeats.insert(repeats.begin(), 1lu);
        }
    }
}

void MKLDNNTileNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW() << errorPrefix << " has incorrect number of input edges. "
                "Expected: 2, Actual: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << errorPrefix << " has no output edges.";
    const auto& dstDims0 = getOutputShapeAtPort(0).getDims();
    for (size_t i = 1; i < getChildEdges().size(); i++) {
        const auto& dstDims = getOutputShapeAtPort(i).getDims();
        if (dstDims.size() != dstDims0.size())
            IE_THROW() << errorPrefix << " has output edges 0 and " << i << " with different ranks.";
        for (size_t j = 0; j < dstDims0.size(); j++) {
            if (dstDims0[j] != dstDims[j]) {
                IE_THROW() << errorPrefix << " has output edges 0 and " << i << " with different dims.";
            }
        }
    }
    if (getInputShapeAtPort(0).getRank() > getOutputShapeAtPort(0).getRank())
        IE_THROW() << errorPrefix << " has incorrect input/output data shape rank. Input shape rank cannot be more than output shape rank. "
                "Actual input shape size: " << getInputShapeAtPort(0).getRank() << ", output shape size: " << getOutputShapeAtPort(0).getRank();
}

void MKLDNNTileNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors = getSupportedConfigs(this);
}

void MKLDNNTileNode::createPrimitive() {
std::cout << "MKLDNNTileNode::createPrimitive" << std::endl;
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

bool MKLDNNTileNode::needPrepareParams() const {
    return MKLDNNNode::needPrepareParams();
}

void MKLDNNTileNode::prepareParams() {
std::cout << "MKLDNNTileNode::prepareParams" << std::endl;
    if (isDynamic) {
        optimizedCase = false;
        return;
    }

    auto srcBlockedDims = getParentEdgeAt(TILE_INPUT)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto dstBlockedDims = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();

    optimizedCase = prepareOptimizedParams(this, srcBlockedDims, dstBlockedDims);
}

void MKLDNNTileNode::execute(mkldnn::stream strm) {
    if (optimizedCase) {
        optimizedExecute(this);
    } else {
        notOptimizedExecute(strm);
    }
}

void MKLDNNTileNode::notOptimizedExecute(mkldnn::stream strm) {
    if (noTiling) {
        return;
    }

    auto& srcMemory = getParentEdgeAt(0)->getMemory();

    const uint8_t* src_ptr = reinterpret_cast<const uint8_t*>(srcMemory.GetPtr());
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetPtr());

    int m_inner_dim = 1;
    int m_outer_dim = 1;
    auto inDims = srcMemory.getStaticDims();
    for (int i=0; i < axis; i++ ) m_outer_dim *= inDims[i];
    for (int i=axis; i < inDims.size(); i++ ) m_inner_dim *= inDims[i];
    if (axis > 0) {
        m_outer_dim /= inDims[0];
        m_outer_dim *= batchToProcess();
    } else {
        m_inner_dim /= inDims[0];
        m_inner_dim *= batchToProcess();
    }

    if (m_inner_dim == 1 && m_outer_dim % 8 == 0 && srcMemory.getDesc().hasLayoutType(LayoutType::nCsp8c)) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw8c)
         */
        m_inner_dim *= 8;
        m_outer_dim /= 8;
    } else if (m_inner_dim == 1 && m_outer_dim % 16 == 0 && srcMemory.getDesc().hasLayoutType(LayoutType::nCsp16c)) {
        /*
         * We may enable tile processing directly to appropriate output format (nChw16c)
         */
        m_inner_dim *= 16;
        m_outer_dim /= 16;
    }

    m_inner_dim *= srcMemory.getDesc().getPrecision().size();
    for (int i = 0; i < m_outer_dim; ++i) {
        for (int t = 0; t < tiles; ++t) {
            cpu_memcpy(dst_ptr, src_ptr, m_inner_dim);
            dst_ptr += m_inner_dim;
        }
        src_ptr += m_inner_dim;
    }
}

bool MKLDNNTileNode::created() const {
    return getType() == Tile;
}

REG_MKLDNN_PRIM_FOR(MKLDNNTileNode, Tile);
