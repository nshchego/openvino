// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <cmath>
#include <vector>
#include <string>
#include <mkldnn_types.h>
#include "ie_parallel.hpp"
#include "utils/bfloat16.hpp"
#include <mkldnn_selective_build.h>
#include "mkldnn_broadcast_node.h"
#include <nodes/common/blocked_desc_creator.h>
#include <ngraph/opsets/opset1.hpp>
#include "common/cpu_memcpy.h"

using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNBroadcastNode::isSupportedOperation(const std::shared_ptr<const ov::Node>& op, std::string& errorMessage) noexcept {
    try {
        if (!one_of(op->get_type_info(),
                ov::op::v1::Broadcast::type_info)) {
            errorMessage = "Only Broadcast operations from opset1 are supported.";
            return false;
        }
        auto broadcastOp = ov::as_type_ptr<const ov::op::v1::Broadcast>(op);
        if (!one_of(broadcastOp->get_broadcast_spec().m_type, ov::op::AutoBroadcastType::NUMPY, ov::op::AutoBroadcastType::EXPLICIT)) {
            errorMessage = "Only NUMPY and EXPLICIT broadcast types are supported.";
            return false;
        }
        if (!isDynamicNgraphNode(op) &&
                (op->get_input_node_ptr(TARGET_SHAPE_IDX)->get_type_info() != ov::op::v0::Constant::type_info ||
                op->get_input_size() > AXES_MAPPING_IDX &&
                op->get_input_node_ptr(AXES_MAPPING_IDX)->get_type_info() != ov::op::v0::Constant::type_info)) {
            errorMessage = "Only constant shape input is supported.";
            return false;
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNBroadcastNode::MKLDNNBroadcastNode(const std::shared_ptr<ov::Node>& op, const mkldnn::engine& eng,
        MKLDNNWeightsSharing::Ptr &cache) : MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        IE_THROW(NotImplemented) << errorMessage;
    }

    errorPrefix = "Broadcast node with name '" + op->get_friendly_name() + "' ";
    if (op->get_input_size() != 2 && op->get_input_size() != 3)
        IE_THROW() << errorPrefix << "has incorrect number of input edges: " << getParentEdges().size();
    if (op->get_output_size() == 0)
        IE_THROW() << errorPrefix << "has no output edges.";

    auto broadcastOp = ov::as_type_ptr<const ov::op::v1::Broadcast>(op);
    if (broadcastOp->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::NUMPY) {
        broadcastType = NUMPY;
    } else if (broadcastOp->get_broadcast_spec().m_type == ov::op::AutoBroadcastType::EXPLICIT) {
        if (op->get_input_size() <= AXES_MAPPING_IDX)
            IE_THROW() << errorPrefix << " and EXPLICIT mode must have tree input edges: " << getParentEdges().size();
        broadcastType = EXPLICIT;
    }

    if (op->get_input_node_ptr(TARGET_SHAPE_IDX)->get_type_info() == ov::op::v0::Constant::type_info) {
        constMap[TARGET_SHAPE_IDX] = true;
    }
    if (broadcastType == EXPLICIT &&
                op->get_input_node_ptr(AXES_MAPPING_IDX)->get_type_info() == ov::op::v0::Constant::type_info) {
        constMap[AXES_MAPPING_IDX] = true;
        auto axesMappingOp = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXES_MAPPING_IDX));
            axesMapping = axesMappingOp->get_vector<int32_t>();
    }

//    if (op->get_input_node_ptr(TARGET_SHAPE_IDX)->get_type_info() == ov::op::v0::Constant::type_info) {
//        constMap[TARGET_SHAPE_IDX] = true;
//        const auto& srcDims = getInputShapeAtPort(INPUT_DATA_IDX).getDims();
//        const auto& dstDims = getOutputShapeAtPort(0).getDims();
//        repeats = dstDims;
//
//        if (broadcastType == NUMPY) {
//            const auto ndims = dstDims.size();
//            for (int i = 0; i < srcDims.size(); i++) {
//                repeats[ndims - 1 - i] /= srcDims[srcDims.size() - 1 - i];
//            }
//        } else if (broadcastType == EXPLICIT &&
//                op->get_input_node_ptr(AXES_MAPPING_IDX)->get_type_info() == ov::op::v0::Constant::type_info) {
//            constMap[AXES_MAPPING_IDX] = true;
//            auto axesMappingOp = ov::as_type<ov::op::v0::Constant>(op->get_input_node_ptr(AXES_MAPPING_IDX));
//            axesMapping = axesMappingOp->get_vector<int32_t>();
//
//            for (size_t i = 0; i < axesMapping.size(); i++) {
//                repeats[axesMapping[i]] /= srcDims[i];
//            }
//        }
//    }
}

void MKLDNNBroadcastNode::getSupportedDescriptors() {
std::cout << "MKLDNNBroadcastNode::getSupportedDescriptors" << std::endl;
    const auto& srcDims = getInputShapeAtPort(INPUT_DATA_IDX).getDims();
    repeats = getOutputShapeAtPort(0).getDims();
    const auto ndims = repeats.size();

    if (broadcastType == NUMPY) {
        for (size_t i = 0lu; i < srcDims.size(); i++) {
            repeats[ndims - 1 - i] /= srcDims[srcDims.size() - 1 - i];
        }
    } else if (broadcastType == EXPLICIT) {
        const size_t axLen = getInputShapeAtPort(AXES_MAPPING_IDX).getDims()[0];
        if (constMap[AXES_MAPPING_IDX]) {
            for (size_t i = 0lu; i < axLen; i++) {
                repeats[axesMapping[i]] /= srcDims[i];
            }
        } else {
            for (size_t i = 0lu; i < axLen; i++) {
                repeats[i] /= srcDims[i]; // TODO add test
            }
        }
    }
}

void MKLDNNBroadcastNode::initSupportedPrimitiveDescriptors() {
std::cout << "MKLDNNBroadcastNode::initSupportedPrimitiveDescriptors" << std::endl;
    if (!supportedPrimitiveDescriptors.empty())
        return;

    supportedPrimitiveDescriptors = getSupportedConfigs(this);
}

void MKLDNNBroadcastNode::createPrimitive() {
    if (inputShapesDefined()) {
        if (needPrepareParams())
            prepareParams();
        updateLastInputDims();
    }
}

bool MKLDNNBroadcastNode::needPrepareParams() const {
    return MKLDNNNode::needPrepareParams();
}

void MKLDNNBroadcastNode::prepareParams() {
    const auto& srcDims = getInputShapeAtPort(INPUT_DATA_IDX).getDims();
    repeats = getOutputShapeAtPort(0).getDims();
    const auto ndims = repeats.size();

    auto srcBlockedDims = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();
    auto dstBlockedDims = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>()->getBlockDims();

    if (broadcastType == NUMPY) {
        for (int i = 0; i < srcDims.size(); i++) {
            repeats[ndims - 1 - i] /= srcDims[srcDims.size() - 1 - i];
        }
    } else if (broadcastType == EXPLICIT) {
        if (!constMap[AXES_MAPPING_IDX]) {
            auto& axesMappingMemory = getParentEdgeAt(AXES_MAPPING_IDX)->getMemory();
            const int32_t* axesMappingPtr = reinterpret_cast<const int32_t*>(axesMappingMemory.GetPtr());

            const size_t axesMappingSize = axesMappingMemory.GetShape().getElementsCount();
            axesMapping.reserve(axesMappingSize);
            axesMapping.assign(axesMappingPtr, axesMappingPtr + axesMappingSize);
        }

        for (size_t i = 0; i < getInputShapeAtPort(AXES_MAPPING_IDX).getDims()[0]; i++) {
            repeats[axesMapping[i]] /= srcDims[i];
        }

        SizeVector newSrcBlockedDims = SizeVector(dstBlockedDims.size(), 1);
        for (size_t i = 0; i < getInputShapeAtPort(AXES_MAPPING_IDX).getDims()[0]; i++) {
            newSrcBlockedDims[axesMapping[i]] = srcBlockedDims[i];
        }
        srcBlockedDims = newSrcBlockedDims;
    }

    optimizedCase = prepareOptimizedParams(this, srcBlockedDims, dstBlockedDims);
}

void MKLDNNBroadcastNode::execute(mkldnn::stream strm) {
    if (optimizedCase) {
        optimizedExecute(this);
    } else {
        notOptimizedExecute(strm);
    }
}

void MKLDNNBroadcastNode::notOptimizedExecute(mkldnn::stream strm) {
std::cout << "MKLDNNBroadcastNode::notOptimizedExecute" << std::endl;
    size_t shape_size = (getParentEdgeAt(TARGET_SHAPE_IDX)->getMemory().getStaticDims())[0];
    SizeVector dst_dims = getChildEdgeAt(0)->getMemory().getStaticDims();
    SizeVector src_dims = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().getStaticDims();

    auto srcDesc = getParentEdgeAt(INPUT_DATA_IDX)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    SizeVector srcStrides = srcDesc->getStrides();
    size_t data_size = srcDesc->getPrecision().size();

    if (!src_dims.size())
        src_dims = SizeVector(1, 1);
    if (!srcStrides.size())
        srcStrides = SizeVector(1, 1);

    if (dst_dims.size() != shape_size) {
        IE_THROW() << "Output tensor dimension mismatch";
    }

    if (src_dims.size() > dst_dims.size()) {
        IE_THROW() << "Output tensor dimension is smaller then input tensor dimension";
    }

    auto dstDesc = getChildEdgeAt(0)->getMemory().GetDescWithType<BlockedMemoryDesc>();
    InferenceEngine::SizeVector dstStrides = dstDesc->getStrides();
    InferenceEngine::SizeVector src_aligned(dst_dims.size());
    InferenceEngine::SizeVector srcStrides_aligned(dst_dims.size());
    size_t prefix_size = dst_dims.size() - src_dims.size();
    for (size_t i = 0; i < dst_dims.size(); i++) {
        if (i < prefix_size) {
            src_aligned[i] = 1;
            srcStrides_aligned[i] = srcStrides[0];
        } else {
            src_aligned[i] = src_dims[i - prefix_size];
            srcStrides_aligned[i] = srcStrides[i - prefix_size];
        }
    }

    size_t work_amount_dst = dstStrides[0] * dst_dims[0];
    const auto *src_data = reinterpret_cast<const uint8_t *>(getParentEdgeAt(INPUT_DATA_IDX)->getMemoryPtr()->GetPtr());
    auto *dst_data = reinterpret_cast<uint8_t *>(getChildEdgeAt(0)->getMemoryPtr()->GetPtr());

    parallel_nt(0, [&](const int ithr, const int nthr) {
        size_t i, src_idx, start = 0, end = 0;
        SizeVector counters(dst_dims.size(), 0);
        splitter(work_amount_dst, nthr, ithr, start, end);
        for (int j = dst_dims.size() - 1, i = start; j >= 0; j--) {
            counters[j] = i % dst_dims[j];
            i /= dst_dims[j];
        }
        for (size_t iwork = start * data_size; iwork < end * data_size; iwork += data_size) {
            for (i = 0, src_idx = 0; i < dst_dims.size(); ++i)
                src_idx += counters[i] ? ((counters[i] % src_aligned[i]) * srcStrides_aligned[i]) : 0;

            cpu_memcpy(&dst_data[iwork], &src_data[src_idx * data_size], data_size);

            for (int j = dst_dims.size() - 1; j >= 0; j--) {
                counters[j] = (counters[j] + 1) % dst_dims[j];
                if (counters[j] != 0) break;
            }
        }
    });
}

bool MKLDNNBroadcastNode::created() const {
    return getType() == Broadcast;
}

REG_MKLDNN_PRIM_FOR(MKLDNNBroadcastNode, Broadcast)
