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
        if (op->get_input_shape(TILE_INPUT).size() != op->get_input_shape(TILE_REPEATS)[0]) {
            errorMessage = "Doesn't support inputs with different ranks";
            return false;
        }
        const auto repeatsNode = std::dynamic_pointer_cast<const ov::op::v0::Constant>(op->get_input_node_shared_ptr(TILE_REPEATS));
        if (repeatsNode == nullptr) {
            errorMessage = "Only const 'repeats' input is supported";
            return false;
        }
        const auto repeats = repeatsNode->cast_vector<int32_t>();
        if (std::count_if(repeats.begin(), repeats.end(), [](int32_t x) { return x > 1; }) > 1) {
            errorMessage = "Doesn't support 'repeats' with more than one specified axis";
            return false;
        }
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

    const auto repeatsNode = std::dynamic_pointer_cast<const ov::op::v0::Constant>(op->get_input_node_shared_ptr(TILE_REPEATS));
    if (!repeatsNode)
        IE_THROW() << errorPrefix << " has non constant second input.";
    const auto repeats = repeatsNode->cast_vector<int32_t>();
    // At this moment CPU plug-in supports tiling only per single axis
    // This behavoiur is guaranteed by ConvertTileToSeqTiles
    for (size_t i = 0; i < repeats.size(); i++) {
        if (repeats[i] > 1) {
            axis = i;
            tiles = repeats[i];
            break;
        }
    }
    noTiling = axis == -1;
    if (axis >= static_cast<int>(op->get_input_shape(TILE_INPUT).size()))
        IE_THROW() << errorPrefix << " has incorrect tiling axis: " << axis;
    if (tiles < 1 && !noTiling)
        IE_THROW() << errorPrefix << " has incorrect 'repeats' value: " << tiles;
}

void MKLDNNTileNode::getSupportedDescriptors() {
    if (getParentEdges().size() != 2)
        IE_THROW() << "Tile node with name " << getName() << " has incorrect number of input edges. "
                "Expected: 2, Actual: " << getParentEdges().size();
    if (getChildEdges().empty())
        IE_THROW() << "Tile node with name " << getName() << " has no output edges.";
    auto dstDims0 = getOutputShapeAtPort(0).getDims();
    for (int i = 1; i < getChildEdges().size(); i++) {
        auto DstDims = getOutputShapeAtPort(i).getDims();
        if (DstDims.size() != dstDims0.size())
            IE_THROW() << "Output edges 0 and " << i << " have different dims for Tile node with name " << getName();
        for (int j = 0; j < dstDims0.size(); j++) {
            if (dstDims0[j] != DstDims[j]) {
                IE_THROW() << "Output edges 0 and " << i << " have different dims for Tile node with name " << getName();
            }
        }
    }
    if (getInputShapeAtPort(0).getDims().size() > getOutputShapeAtPort(0).getDims().size())
        IE_THROW() << "Tile node with name " << getName() << " has incorrect input shape. Input shape cannot be more than output shape. "
                "Actual input shape size: " << getInputShapeAtPort(0).getDims().size() << ", output shape size: " << getOutputShapeAtPort(0).getDims().size();

    if (getInputShapeAtPort(1).getDims().size() != 1)
        IE_THROW() << "Repeats must be 1D tensor for Tile node with name " << getName();
    if (!getParentEdgeAt(1)->getParent()->isConstant())
        IE_THROW() << "Tile node with name " << getName() << " has non constant parent Node on 1-st input. "
                "This case is currently not supported in CPU plug-in.";

//    auto& srcMemory = getParentEdgeAt(1)->getMemory();
//    const int32_t* repeatsPtr = reinterpret_cast<const int32_t*>(srcMemory.GetPtr());
//
//    size_t repeatsSize = getParentEdgeAt(1)->getMemory().GetShape().getElementsCount();
//    std::vector<int> repeatsData(repeatsSize);
//    for (int i = 0; i < repeatsSize; i++) {
//        repeatsData[i] = repeatsPtr[i];
//    }
//
//    for (int i = 0; i < getInputShapeAtPort(1).getDims()[0]; i++) {
//        repeats.push_back(repeatsData[i]);
//    }
//    while (repeats.size() < getOutputShapeAtPort(0).getDims().size()) {
//        repeats.insert(repeats.begin(), 1);
//    }
}

void MKLDNNTileNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    auto& srcMemory = getParentEdgeAt(TILE_REPEATS)->getMemory();
    const int32_t* repeatsPtr = reinterpret_cast<const int32_t*>(srcMemory.GetPtr());

    size_t repeatsSize = getParentEdgeAt(1)->getMemory().GetShape().getElementsCount();
    std::vector<int> repeatsData(repeatsSize);
    for (int i = 0; i < repeatsSize; i++) {
        repeatsData[i] = repeatsPtr[i];
    }

    for (int i = 0; i < getInputShapeAtPort(TILE_REPEATS).getDims()[0]; i++) {
        repeats.push_back(repeatsData[i]);
    }
    while (repeats.size() < getOutputShapeAtPort(0).getDims().size()) {
        repeats.insert(repeats.begin(), 1);
    }

    supportedPrimitiveDescriptors = getSupportedConfigs(this);
}

void MKLDNNTileNode::createPrimitive() {
    for (int i = 0; i < getChildEdges().size(); i++) {
        auto& dstMemPtr = getChildEdgeAt(i)->getMemoryPtr();
        if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
            IE_THROW() << "Destination memory " << i << "didn't allocate for Tile node with name " << getName();
    }
    for (int i = 0; i < getParentEdges().size(); i++) {
        auto& srcMemPtr = getParentEdgeAt(i)->getMemoryPtr();
        if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
            IE_THROW() << "Input memory " << i << "didn't allocate for Tile node with name " << getName();
    }
    if (getSelectedPrimitiveDescriptor() == nullptr)
        IE_THROW() << "Preferable primitive descriptor is not set for Tile node with name " << getName();

    SizeVector srcBlockedDims = getParentEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    SizeVector dstBlockedDims = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
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
